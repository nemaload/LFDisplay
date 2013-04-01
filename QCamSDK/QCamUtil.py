"""
Various helpful utilities for use with QCam
"""

import QCam
import Queue
import ctypes
import threading
import gc
import struct

class Error(Exception):
    pass

class CameraQueue:
    """
    A frame queue for keeping track of frames in flight from the camera
    """

    Empty = Queue.Empty

    def __init__(self, camera):
        """
        Create a new FrameQueue object

        camera - An opened QCam camera
        size - The maximum number of frames in flight
        """
        self._queue = Queue.Queue(0)
        self.camera = camera
        self.lock = threading.RLock()
        self.frames = {} # frames indexed by pBuffer
        self.streaming = False
        self.paused = False
        self.callback = QCam.AsyncCallback(self._frame_arrived)

    def start(self, size=5):
        """Start streaming from the camera"""
        # clear the queue
        self.camera.Abort()
        # prepare the camera for stream capture by turning on streaming
        self.camera.StartStreaming()
        self.lock.acquire()
        self.streaming = True
        self.lock.release()
        # start the frames in flight
        if size < 2:
            raise Error('Must have at least two frames in flight')
        for i in range(size):
            self.put()

    def stop(self):
        """Stop streaming from the camera"""
        self.camera.Abort()
        self.camera.StopStreaming()
        self.lock.acquire()
        self.streaming = False
        self.frames = {} # remove pointers to the frames
        self.lock.release()
        # garbage collect the frame pointers
        gc.collect()

    def active(self):
        "Return whether streaming is currently active"
        self.lock.acquire()
        streaming = self.streaming
        self.lock.release()
        return streaming

    def pause(self):
        "Abort all the current frames in progress"
        self.lock.acquire()
        paused = self.paused
        self.lock.release()
        if paused:
            return # already paused
        self.camera.Abort()
        self.camera.StopStreaming()
        self.lock.acquire()
        self.paused = True
        self.lock.release()

    def unpause(self):
        "Recover from a pause"
        self.lock.acquire()
        paused = self.paused
        self.lock.release()
        if not paused:
            return # not paused
        # start streaming again
        self.camera.StartStreaming()
        self.lock.acquire()
        # queue up the previously queued frames
        frames = self.frames.values()
        self.lock.release()
        for frame in frames:
            self.put(frame)
        self.lock.acquire()
        self.paused = False
        self.lock.release()

    def paused(self):
        self.lock.acquire()
        paused = self.paused
        self.lock.release()
        return paused

    def __del__(self):
        # shut down camera streaming
        self.stop()

    def put(self, frame=None):
        """Add a frame, ready for capture, into the queue for the camera"""
        self.lock.acquire()
        streaming = self.streaming
        self.lock.release()
        if not streaming:
            return # discard the frame if not streaming
        # the gain-exposure product will determine relative intensity
        gainExposureFloat = 0.000001 * self.camera.settings.exposure * 0.000001 * self.camera.settings.normalizedGain
        gainExposure, = struct.unpack('L',struct.pack('f',gainExposureFloat))
        frame = self.camera.QueueFrame(self.callback, frame, gainExposure)
        # keep a reference to this frame so that it doesn't get deleted
        self.lock.acquire()
        if frame.pBuffer not in self.frames:
            self.frames[frame.pBuffer] = frame
        self.lock.release()

    def get(self, block=True, timeout=None):
        """Get a captured frame from the queue"""
        return self._queue.get(block,timeout)

    def frame_done(self):
        """Called to indicate that a frame is ready"""
        pass

    def _frame_arrived(self, pointer, data, error, flags):
        """Process a freshly arrived frame"""
        frame = ctypes.cast(pointer, ctypes.POINTER(QCam.Frame)).contents
        if error != QCam.qerrSuccess:
            raise QCam.Error(error)
        # recover the string buffer
        BufferType = ctypes.c_char * frame.bufferSize
        frame.stringBuffer = BufferType.from_address(frame.pBuffer)
        frame.formatString = QCam.image_fmt_to_string(frame.format)
        
        gainExposureFloat, = struct.unpack('f',struct.pack('L',data))
        frame.intensity = gainExposureFloat # normalized intensity of the frame
        
        # stick it into the output queue
        self._queue.put(frame)
        # frame ready
        self.frame_done()

