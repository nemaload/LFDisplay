"""
Input modules for handling file input
"""

from PyQt4 import QtCore, QtGui
import Queue
import os
import numpy
import struct
import threading
import thread
import time
import sys

import gui

DEBUG = False

import _FreeImage

class Error(Exception):
    pass

class InputFrame:
    pass

def _load_frames(filepath, max_frames=0):
    """
    Load frame(s) from a single file

    max_frames is the maximum number of frames to load

    returns a list of frames
    """
    frame_list = []
    # now attempt to load the file
    if filepath[-4:].lower() == '.tmp':
        # ImageStack tmp file
        f=open(filepath,'rb')
        frames, width, height, channels = struct.unpack('iiii',f.read(16))
        if frames < 1:
            raise Error('TMP files must have at least one frame')
        if channels == 1:
            formatString = 'mono16'
        elif channels == 3:
            formatString = 'rgb48'
        else:
            raise Error('Unsupported number of channels: '+str(channels))
        # read the data
        if max_frames:
            # if max_frames is specified, load only up to that many frames
            frames = min(max_frames,frames)
        for frame_no in frames:
            data = f.read(width*height*channels*4)
            if len(data) != width*height*channels*4:
                break # premature end of file
            # create a numpy array
            arr = numpy.fromstring(data,dtype='float32')
            # convert to 16bit
            arr = (65536*arr).astype('uint16')
            # make the frame
            cur_frame = InputFrame()
            cur_frame.formatString = formatString
            cur_frame.stringBuffer = arr.tostring()
            cur_frame.width, cur_frame.height = width, height
            cur_frame.bits = 16
            cur_frame.channels = channels
            cur_frame.intensity = 1.0
            frame_list.append(cur_frame)
        f.close()
    else: # try using FreeImagePy
        cur_frame = InputFrame()
        # determine the filetype of the file
        fif = _FreeImage.GetFileType(filepath)
        if fif == _FreeImage.FIF_UNKNOWN:
            # try to guess it from the filename
            fif = _FreeImage.GetFIFFromFilename(filepath)
        if fif == _FreeImage.FIF_UNKNOWN:
            raise Error("Unable to determine file format of "+filepath)

        # load the file into a FreeImage bitmap
        dib = _FreeImage.Load(fif, filepath)

        width = _FreeImage.GetWidth(dib)
        height = _FreeImage.GetHeight(dib)
        bpp = _FreeImage.GetBPP(dib)
        imagetype = _FreeImage.GetImageType(dib)
        pitch = _FreeImage.GetPitch(dib)
        data = _FreeImage.ConvertToRawBits(dib, -1, _FreeImage.GetBPP(dib), 0, 0, 0, topdown=True)
        colortype = _FreeImage.GetColorType(dib)
        _FreeImage.Unload(dib)

        if imagetype == _FreeImage.FIT_BITMAP and bpp == 8 and colortype == _FreeImage.FIC_MINISBLACK:
            # Monochrome 8-bit data
            cur_frame.stringBuffer = data
            cur_frame.width, cur_frame.height = width, height
            cur_frame.formatString = 'mono8'
            cur_frame.bits = 8
            cur_frame.channels = 1
        elif imagetype == _FreeImage.FIT_UINT16:
            # Monochrome 16-bit data
            cur_frame.stringBuffer = data
            cur_frame.width, cur_frame.height = width, height
            cur_frame.formatString = 'mono16'
            cur_frame.bits = 16
            cur_frame.channels = 1
        elif imagetype == _FreeImage.FIT_BITMAP and bpp == 24:
            # Color 8-bit-per-channel data
            cur_frame.stringBuffer = data
            cur_frame.width, cur_frame.height = width, height
            # FreeImage as of 2007/07 treats Apple platform as big endian
            # when looking at 24bit or 32bit color values
            if sys.platform == 'darwin' or sys.byteorder != 'little':
                cur_frame.formatString = 'rgb24'
            else:
                cur_frame.formatString = 'bgr24'
            cur_frame.bits = 8
            cur_frame.channels = 3
        elif imagetype == _FreeImage.FIT_BITMAP and bpp == 32:
            # Color 8-bit-per-channel data plus alpha
            cur_frame.stringBuffer = data
            cur_frame.width, cur_frame.height = width, height
            # FreeImage as of 2007/07 treats Apple platform as big endian
            # when looking at 24bit or 32bit color values
            if sys.platform == 'darwin' or sys.byteorder != 'little':
                cur_frame.formatString = 'rgba32'
            else:
                cur_frame.formatString = 'bgra32'
            cur_frame.bits = 8
            cur_frame.channels = 4
        elif imagetype == _FreeImage.FIT_RGB16:
            # Color 16-bit-per-channel data
            cur_frame.stringBuffer = data
            cur_frame.width, cur_frame.height = width, height
            cur_frame.formatString = 'rgb48'
            cur_frame.bits = 16
            cur_frame.channels = 3
        elif imagetype == _FreeImage.FIT_RGBA16:
            # Color+alpha 16-bit-per-channel data
            cur_frame.stringBuffer = data
            cur_frame.width, cur_frame.height = width, height
            cur_frame.formatString = 'rgba64'
            cur_frame.bits = 16
            cur_frame.channels = 4
        else:
            print imagetype, bpp, colortype, width, height
            raise Error('Unsupported image mode')
        # relative intensity
        cur_frame.intensity = 1.0
        frame_list.append(cur_frame)
    if DEBUG:
        print 'frame loaded'
        print imagetype
        print bpp
        print cur_frame.width, cur_frame.height
        print cur_frame.bits, cur_frame.formatString
        print len(cur_frame.stringBuffer)
    return frame_list

import inputs

class SingleFileQueue:
    "A queue that only has one frame in it"

    # all queues must have an Empty exception as a member
    Empty = Queue.Empty
    
    def __init__(self, filepath, frame):
        """
        Initialize the given input, etc.  Other queues may need extra
        arguments, like the camera number, etc.
        """
        self.filepath = filepath
        self._streaming = False
        # load the first frame from a file
        self._frame = frame

    def start(self):
        """
        Start streaming frames to the receivers

        Since there is only one frame, this causes the single
        frame to be pushed out to receivers
        """
        self._streaming = True
        self.frame_done()

    def stop(self):
        """
        Stop streaming frames to the receivers

        Since there is only one frame, this function does nothing
        but set a flag
        """
        self._streaming = False

    def active(self):
        """
        Return whether frames are currently streaming
        """
        return self._streaming

    def pause(self):
        """
        Temporarily pause the streaming of frames
        """
        self._paused = True

    def unpause(self):
        """
        Undo a pause
        """
        self._paused = False

    def paused(self):
        """
        Return whether we are currently paused
        """
        return self._paused

    def __del__(self):
        """
        The destructor should attempt to stop the queue
        """
        self.stop()
        self.frame_done = None
        self._frame = None

    def put(self, frame=None):
        """
        Add a frame, ready for capture, into the queue
        If None, a frame will be created.

        Normally, this should be done after the frame has
        been used by the receiver and this call indicates
        that the frame is ready for reuse

        Since there's only one frame, doesn't really matter
        """
        pass

    def get(self, block=True, timeout=None):
        """
        Return a ready-to-read frame from the queue
        This function should wait for the specified timeout
        or indefinitely (if timeout is None) if block is True
        If block is False, or if no more frames are expected,
        raise the Empty exception
        """
        return self._frame

    def frame_done(self):
        """
        This function is called when a frame is ready on
        the queue.  Normally this is set to a different
        function instead of this dummy function by the receiver
        so that the receiver can get notifications
        """
        pass

    def set_frame(self, frame):
        "Set a new frame for the single frame input"
        self._frame = frame

class SingleFileInput(inputs.OpenedInput):
    "A modified version of OpenedInput that forces only one frame to be shown at a time"

    def __init__(self, queue, widget):
        inputs.OpenedInput.__init__(self, queue, widget, True)

    def setStreaming(self, streaming):
        """
        Change whether we are streaming
        """
        oldStreaming = self._queue.active()
        if oldStreaming != streaming:
            if streaming:
                self.setCount(1)
                self._queue.start()
                self.emit(QtCore.SIGNAL('desiredIntensityChanged(float'),1.0)
            else:
                self._queue.stop()
            # can't use local variable streaming here because streaming state may have changed
            self.emit(QtCore.SIGNAL('streamingChanged(bool)'),self._queue.active())

class SingleFileSettings(QtGui.QWidget):
    """
    A window that has the various image info
    """
    def __init__(self, queue, settings, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.queue = queue
        self.settings = settings

        self.title = gui.TitleWidget(label='',
                                     title='Single Image',
                                     description='Load a single image file as a light field')

        pathname,filename = os.path.split(queue.filepath)

        props = { 'Image Width':queue._frame.width,
                  'Image Height':queue._frame.height,
                  'Image Bit-depth':queue._frame.bits,
                  'Location':pathname,
                  'Filename':filename}

        self.properties = gui.PropertiesWidget(props,
                                               title='Image Properties',
                                               headers=None,
                                               )

        self.settingsLayout = QtGui.QGridLayout(self)
        self.settingsLayout.addWidget(self.title,0,0)
        self.settingsLayout.addWidget(self.properties,1,0)
        self.settingsLayout.setRowStretch(4,1)
        self.setLayout(self.settingsLayout)

    def newFrame(self):
        pass

    def setStreaming(self, streaming):
        pass

    def setCount(self, count):
        pass

class SingleFileInputPlugin:
    "Input plugin to load a single file"

    name = "Single image file..."
    description = "Load a single image file as a light field."

    def __init__(self):
        pass

    def get_input(self, parent):
        """
        Display an input selection dialog if necessary and return an
        OpenedInput object

        Any exceptions raised would cause an error box to pop up
        """
        input_path = parent.settings.getString('input/file/default_path','')
        filepath = QtGui.QFileDialog.getOpenFileName(parent,
                                                     'Choose an image file to load as a light field',
                                                     input_path,
                                                     'Images (*.*)')
        if filepath:
            filepath = str(filepath) # convert to Python string
            # save our input path
            new_input_path = os.path.split(filepath)[0]
            parent.settings.setValue('input/file/default_path',new_input_path)
            # load that single frame
            progress = QtGui.QProgressDialog('Loading %s...' % filepath,
                                             'Cancel',
                                             0,
                                             1,
                                             parent)
            progress.setCancelButton(None)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.show()
            try:
                QtGui.qApp.processEvents()
                progress.setValue(0)
                frame = _load_frames(filepath, 1)[0]
            finally:
                progress.setValue(1)
                progress.hide()
                QtGui.qApp.processEvents()
            progress.setValue(1)
            progress.hide()
            QtGui.qApp.processEvents()
            # initialize the queue
            queue = SingleFileQueue(filepath, frame)
            widget = SingleFileSettings(queue, parent.settings)
            return SingleFileInput(queue, widget)
        else:
            return None
        
    def close_input(self, opened_input):
        """
        Close an input that was previously opened by get_input

        The queue and widget correspond to the return values of the
        get_input() method
        """
        del opened_input

class MultipleFileQueue:
    "A queue that can have multiple frames in it"

    # all queues must have an Empty exception as a member
    Empty = Queue.Empty
    
    def __init__(self):
        """
        Initialize the given input, etc.  Other queues may need extra
        arguments, like the camera number, etc.
        """
        # lookup table from path to frame data
        self._frame_data = {}
        # the sequence of frames
        # a list of tuples (framename,filepath,framenumber)
        self._frame_sequence = []
        # the current frame number
        self._cur_frame = 0
        # state
        self._streaming = False
        self._paused = False
        self._lock = threading.RLock() # to keep the state consistent
        self._frame_delay = 100 # amount of time per frame, in ms
        # the outgoing queue
        self._queue = Queue.Queue()

    def stream_thread(self):
        "The thread used for streaming"
        self._lock.acquire()
        self._streaming = True
        frame_data = self._frame_data
        self._lock.release()
        while self.active():
            # if paused, pause here until unpaused (polling mode)
            while self.paused():
                time.sleep(0.1)
                if not self.active():
                    break
            if not self.active():
                break
            # change frame number if necessary and grab a frame
            self._lock.acquire()
            frame_sequence = self._frame_sequence
            if not frame_sequence:
                self._lock.release()
                time.sleep(0.1)
                continue # no frames, so sleep and continue running
            self._cur_frame = self._cur_frame % len(frame_sequence)
            cur_frame = self._cur_frame
            self._cur_frame = (self._cur_frame+1)%len(frame_sequence)
            frame_delay = self._frame_delay
            self._lock.release()
            try:
                framename, filepath, frameno = frame_sequence[cur_frame]
                frame = frame_data[filepath][1][frameno]
                # put the frame into the queue
                self._queue.put(frame)
                # notify listeners
                self.frame_done()
                self.cur_frame_changed(cur_frame)
            except Exception, e:
                print >> sys.stderr, 'Unable to display frame'
                print >> sys.stderr, e
            # sleep for the required amount of time
            while frame_delay > 1000:
                time.sleep(1.0)
                if not self.active():
                    break
                frame_delay -= 1000
            if not self.active():
                break
            time.sleep(0.001*frame_delay)
 
    def load_files(self, frame_paths, update=None):
        "Load files and return a frame sequence of the data in those files"
        new_frame_sequence = []
        count = 0
        for filepath in frame_paths:
            framelist = None
            self._lock.acquire()
            if filepath in self._frame_data:
                # already loaded
                framebasename,framelist = self._frame_data[filepath]
            else:
                # see how many other files were named this
                framebasename = os.path.basename(filepath)
                otherbasenames = [os.path.basename(x) for x in self._frame_data.keys() if os.path.basename(x) == framebasename]
                if otherbasenames:
                    framebasename += "(%d)" % (len(otherbasenames)+1)
            self._lock.release()
            if not framelist:
                try:
                    framelist = _load_frames(filepath)
                    if not framelist:
                        raise Error('No frames loaded')
                except Exception, e:
                    print >> sys.stderr, "Skipping "+filepath+"..."
                    print >> sys.stderr, e
                    continue
            # now that we have a frame list, add them
            self._lock.acquire()
            self._frame_data[filepath] = (framebasename,framelist)
            self._lock.release()
            for i in range(len(framelist)):
                framename = framebasename
                if len(framelist) > 1:
                    framename += ':' + str(i+1)
                new_frame_sequence.append((framename,filepath,i))
            count += 1
            if update: # call progress callback
                canceled = update(count)
                if canceled:
                    break
        return new_frame_sequence

    def trim(self):
        "Remove unused files from the frame data cache"
        self._lock.acquire()
        needed = [y for (x,y,z) in self._frame_sequence]
        unneeded = [x for x in self._frame_data.keys() if x not in needed]
        for x in unneeded:
            del self._frame_data[x]
        self._lock.release()
    
    def get_frame_sequence(self):
        "Return the current frame sequence"
        self._lock.acquire()
        frame_sequence = self._frame_sequence[:]
        self._lock.release()
        return frame_sequence

    def set_frame_sequence(self, frame_sequence):
        "Set a new frame sequence"
        self._lock.acquire()
        self._frame_sequence = frame_sequence[:]
        self._lock.release()

    def get_frame_delay(self):
        "Return the current frame delay time"
        self._lock.acquire()
        frame_delay = self._frame_delay
        self._lock.release()
        return frame_delay

    def set_frame_delay(self, frame_delay):
        "Set a new frame delay time"
        self._lock.acquire()
        self._frame_delay = frame_delay
        self._lock.release()

    def get_frame_number(self):
        "Return the current frame number"
        self._lock.acquire()
        frame_number = self._cur_frame
        self._lock.release()
        return frame_number

    def set_frame_number(self, frame_number):
        "Set a new frame number"
        self._lock.acquire()
        self._cur_frame = frame_number
        self._lock.release()

    def start(self):
        """
        Start streaming frames to the receivers
        """
        self._lock.acquire()
        self._streaming = True
        self._lock.release()
        thread.start_new_thread(self.stream_thread,())

    def stop(self):
        """
        Stop streaming frames to the receivers
        """
        self._lock.acquire()
        self._streaming = False
        self._lock.release()

    def active(self):
        """
        Return whether frames are currently streaming
        """
        self._lock.acquire()
        streaming = self._streaming
        self._lock.release()
        return streaming

    def pause(self):
        """
        Temporarily pause the streaming of frames
        """
        self._lock.acquire()
        self._paused = True
        self._lock.release()

    def unpause(self):
        """
        Undo a pause
        """
        self._lock.acquire()
        self._paused = False
        self._lock.release()

    def paused(self):
        """
        Return whether we are currently paused
        """
        self._lock.acquire()
        paused = self._paused
        self._lock.release()
        return paused

    def __del__(self):
        """
        The destructor should attempt to stop the queue
        """
        self.stop()
        self.frame_done = None
        self.cur_frame_changed = None
        self._frame_data = None

    def clear(self):
        "Delete all frame data"
        self.set_frame_sequence([])
        self._lock.acquire()
        self._frame_data = {}
        self._lock.release()

    def put(self, frame=None):
        """
        Add a frame, ready for capture, into the queue
        If None, a frame will be created.

        Normally, this should be done after the frame has
        been used by the receiver and this call indicates
        that the frame is ready for reuse

        Since we have a fixed number of frames, doesn't matter
        """
        pass

    def get(self, block=True, timeout=None):
        """
        Return a ready-to-read frame from the queue
        This function should wait for the specified timeout
        or indefinitely (if timeout is None) if block is True
        If block is False, or if no more frames are expected,
        raise the Empty exception
        """
        return self._queue.get(block,timeout)

    def frame_done(self):
        """
        This function is called when a frame is ready on
        the queue.  Normally this is set to a different
        function instead of this dummy function by the receiver
        so that the receiver can get notifications
        """
        pass

    def cur_frame_changed(self, new_frame_number):
        """
        This function is called when the frame number is changed
        """
        pass

class QueueItemModel(QtCore.QAbstractTableModel):
    """
    An item model that returns information about
    the sequence of frames in the queue
    """
    def __init__(self, queue, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self.queue = queue
        self.frameSequence = self.queue.get_frame_sequence()

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation==QtCore.Qt.Vertical:
                return QtCore.QVariant(section)
            else:
                return QtCore.QVariant("Frame")
        return QtCore.QVariant()
    
    def parent(self, index):
        return QtCore.QModelIndex() # no parents

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 1

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.frameSequence)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.column() >= self.columnCount() or index.column() < 0 or index.row() >= self.rowCount() or index.row() < 0:
            return QtCore.QVariant()
        if role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.frameSequence[index.row()][0])
        return QtCore.QVariant()

    def updateFromQueue(self):
        "Update information from the queue"
        self.frameSequence = self.queue.get_frame_sequence()
        self.emit(QtCore.SIGNAL('layoutChanged()'))

class MultipleFileSettings(QtGui.QWidget):
    """
    A window that has information on frames in a sequence
    """
    def __init__(self, queue, settings, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.queue = queue
        self.settings = settings

        self.title = gui.TitleWidget(label='',
                                     title='Image File Sequence',
                                     description='Display a sequence of input images as light fields')

        self.frameListGroup = QtGui.QGroupBox('Image frames',self)

        self.frameListModel = QueueItemModel(self.queue)

        self.frameListWidget = QtGui.QTableView(self)
        self.frameListWidget.setModel(self.frameListModel)
        self.frameListWidget.horizontalHeader().setStretchLastSection(True)
        self.frameListWidget.horizontalHeader().setVisible(False)

        self.moveUpButton = QtGui.QPushButton('Move up')
        self.connect(self.moveUpButton, QtCore.SIGNAL('clicked(bool)'),
                     self.moveUp)
        self.moveDownButton = QtGui.QPushButton('Move down')
        self.connect(self.moveDownButton, QtCore.SIGNAL('clicked(bool)'),
                     self.moveDown)
        self.removeButton = QtGui.QPushButton('Remove')
        self.connect(self.removeButton, QtCore.SIGNAL('clicked(bool)'),
                     self.remove)
        self.addButton = QtGui.QPushButton('Add...')
        self.connect(self.addButton, QtCore.SIGNAL('clicked(bool)'),
                     self.doAddFiles)

        self.frameListLayout = QtGui.QGridLayout(self.frameListGroup)
        self.frameListLayout.addWidget(self.frameListWidget, 0, 0, 1, 4)
        self.frameListLayout.addWidget(self.moveUpButton, 1, 0)
        self.frameListLayout.addWidget(self.moveDownButton, 1, 1)
        self.frameListLayout.addWidget(self.removeButton, 1, 2)
        self.frameListLayout.addWidget(self.addButton, 1, 3)
        self.frameListGroup.setLayout(self.frameListLayout)



        self.frameSliderGroup = QtGui.QGroupBox('Frame control')

        self.frameLabel = QtGui.QLabel('Current frame: ')
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal,self)
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(self.frameListModel.rowCount()-1)
        self.frameNumber = QtGui.QLabel('0000')

        self.connect(self.frameListModel, QtCore.SIGNAL('layoutChanged()'),
                     self.updateSlider)

        self.frameRate = gui.CustomOptionSelectorWidget('Framerate:',
                                                         [('4 fps',4.0),
                                                          ('10 fps',10.0),
                                                          ('24 fps',24.0),
                                                          ('30 fps',30.0)],
                                                         'Custom', float,
                                                         self)
        self.frameRate.setValue(10.0)


        self.frameSliderLayout = QtGui.QGridLayout(self.frameSliderGroup)
        self.frameSliderLayout.addWidget(self.frameLabel, 0, 0)
        self.frameSliderLayout.addWidget(self.frameSlider, 0, 1)
        self.frameSliderLayout.addWidget(self.frameNumber, 0, 2)
        self.frameSliderLayout.addWidget(self.frameRate, 1, 0, 1, 3)
        self.frameSliderLayout.setColumnStretch(1, 1)
        self.frameSliderGroup.setLayout(self.frameSliderLayout)

        self.connect(self.frameRate, QtCore.SIGNAL('valueChanged()'),
                     self.frameRateChanged)

        
        self.queue.cur_frame_changed = self.curFrameChanged
        self.connect(self, QtCore.SIGNAL('frameNumberChanged(int)'),
                     self.frameSlider.setValue)
        self.connect(self.frameSlider, QtCore.SIGNAL('sliderMoved(int)'),
                     self.changeCurrentFrame)
        self.connect(self.frameSlider, QtCore.SIGNAL('sliderPressed()'),
                     self.changeCurrentFrame)


        self.buttonGroup = QtGui.QGroupBox()
        self.buttonGroupLayout = QtGui.QHBoxLayout()
        self.loadButton = QtGui.QPushButton("Load...")
        self.saveButton = QtGui.QPushButton("Save...")
        self.buttonGroupLayout.addStretch(1)
        self.buttonGroupLayout.addWidget(self.loadButton)
        self.buttonGroupLayout.addWidget(self.saveButton)
        self.buttonGroup.setLayout(self.buttonGroupLayout)

        self.connect(self.loadButton, QtCore.SIGNAL('clicked(bool)'),
                     self.loadSequence)
        self.connect(self.saveButton, QtCore.SIGNAL('clicked(bool)'),
                     self.saveSequence)

        self.startupGroup = QtGui.QGroupBox('On startup...')
        self.startupGroupLayout = QtGui.QVBoxLayout()
        self.startupGroup.setLayout(self.startupGroupLayout)
        self.startupButtons = [None,None,None,None]
        self.startupMapper = QtCore.QSignalMapper(self)
        self.startupButtons[1] = QtGui.QRadioButton('Create a new sequence from images...')
        self.startupButtons[2] = QtGui.QRadioButton('Load a previously saved sequence...')
        self.startupButtons[3] = QtGui.QRadioButton('Create a new blank sequence')
        self.startupButtons[0] = QtGui.QRadioButton('Ask me...')
        for i in range(4):
            self.startupMapper.setMapping(self.startupButtons[i], i)
            self.connect(self.startupButtons[i], QtCore.SIGNAL('clicked(bool)'),
                         self.startupMapper, QtCore.SLOT('map()'))
        self.startupGroupLayout.addWidget(self.startupButtons[1])
        self.startupGroupLayout.addWidget(self.startupButtons[2])
        self.startupGroupLayout.addWidget(self.startupButtons[3])
        self.startupGroupLayout.addWidget(self.startupButtons[0])
        self.connect(self.startupMapper, QtCore.SIGNAL('mapped(int)'),
                     self.saveStartupSettings)
        startup = self.settings.getInteger('input/file/multistartup',0)
        if startup < 0 or startup > 3:
            startup = 0
        self.startupButtons[startup].click()
                                               
        self.settingsLayout = QtGui.QGridLayout(self)
        self.settingsLayout.addWidget(self.title,0,0)
        self.settingsLayout.addWidget(self.frameListGroup,1,0)
        self.settingsLayout.addWidget(self.frameSliderGroup,2,0)
        self.settingsLayout.addWidget(self.buttonGroup,3,0)
        self.settingsLayout.addWidget(self.startupGroup,4,0)

        self.settingsLayout.setRowStretch(5,1)
        self.setLayout(self.settingsLayout)

    def saveStartupSettings(self, i):
        self.settings.setValue('input/file/multistartup', i)

    def updateSlider(self):
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(self.frameListModel.rowCount()-1)

    def frameRateChanged(self):
        frameRate = self.frameRate.value()
        self.queue.set_frame_delay(1000.0/frameRate)

    def changeCurrentFrame(self, frameNumber=None):
        if frameNumber is None:
            frameNumber = self.frameSlider.value()
        self.queue.set_frame_number(frameNumber)
        self.emit(QtCore.SIGNAL('countChanged(int)'),1)
        self.emit(QtCore.SIGNAL('streamingChanged(bool)'),True)

    def curFrameChanged(self, frameNumber):
        self.frameNumber.setText('%d' % (frameNumber))
        if self.frameSlider.value() != frameNumber:
            self.emit(QtCore.SIGNAL('frameNumberChanged(int)'), frameNumber)

    def selectedRows(self):
        return [x.row() for x in self.frameListWidget.selectionModel().selectedRows()]

    def moveUp(self):
        "Move the currently selected entry up"
        sequence = self.queue.get_frame_sequence()
        selectedRows = sorted(self.selectedRows())
        selectionMask = [(x in selectedRows) for x in range(len(sequence))] 
        # remove the rows that are already contiguously on top
        nonTop = 0
        while selectedRows and nonTop == selectedRows[0]:
            del selectedRows[0]
            nonTop += 1
        # now bubble all the others up
        for index in selectedRows:
            temp = sequence[index-1]
            sequence[index-1] = sequence[index]
            sequence[index] = temp
            temp = selectionMask[index-1]
            selectionMask[index-1] = selectionMask[index]
            selectionMask[index] = temp
        # set the new sequence
        self.queue.set_frame_sequence(sequence)
        self.frameListModel.updateFromQueue()
        # set the new selection
        selectionModel = self.frameListWidget.selectionModel()
        selectionModel.clearSelection()
        newSelection = [x for x in range(len(sequence)) if selectionMask[x]]
        for x in newSelection:
            selectionModel.select(self.frameListModel.index(x,0),
                                  selectionModel.Select)

    def moveDown(self):
        "Move the currently selected entry down"
        sequence = self.queue.get_frame_sequence()
        selectedRows = sorted(self.selectedRows(), reverse=True)
        selectionMask = [(x in selectedRows) for x in range(len(sequence))] 
        # remove the rows that are already contiguously on bottom
        nonBottom = len(sequence)-1
        while selectedRows and nonBottom == selectedRows[0]:
            del selectedRows[0]
            nonBottom -= 1
        # now bubble all the others down
        for index in selectedRows:
            temp = sequence[index+1]
            sequence[index+1] = sequence[index]
            sequence[index] = temp
            temp = selectionMask[index+1]
            selectionMask[index+1] = selectionMask[index]
            selectionMask[index] = temp
        # set the new sequence
        self.queue.set_frame_sequence(sequence)
        self.frameListModel.updateFromQueue()
        # set the new selection
        selectionModel = self.frameListWidget.selectionModel()
        selectionModel.clearSelection()
        newSelection = [x for x in range(len(sequence)) if selectionMask[x]]
        for x in newSelection:
            selectionModel.select(self.frameListModel.index(x,0),
                                  selectionModel.Select)

    def remove(self):
        "Remove the currently selected entry/entries"
        sequence = self.queue.get_frame_sequence()
        selectedRows = self.selectedRows()
        newIndexes = [x for x in range(len(sequence)) if x not in selectedRows]
        self.queue.set_frame_sequence([sequence[x] for x in newIndexes])
        self.queue.trim()
        self.frameListModel.updateFromQueue()
        self.frameListWidget.selectionModel().clearSelection()

    def doAddFiles(self):
        "Wrapper for addFiles to ignore the boolean argument from the signal"
        return self.addFiles()
        
    def addFiles(self, parent=None):
        "Pop up a GUI to add files"
        if parent is None:
            parent = self
        input_path = self.settings.getString('input/file/default_path','')
        filepaths = QtGui.QFileDialog.getOpenFileNames(parent,
                                                      'Choose image file(s) to add to a light field movie',
                                                     input_path,
                                                     'Images (*.*)')
        if filepaths:
            filelist = [str(filepath) for filepath in filepaths]
            new_input_path = os.path.split(filelist[0])[0]
            self.settings.setValue('input/file/default_path',new_input_path)


            progress = QtGui.QProgressDialog('Loading image sequence...',
                                             'Skip rest of images',
                                             0,
                                             len(filelist),
                                             parent)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.show()
            progress.setValue(0)
            QtGui.qApp.processEvents()

            def update(count):
                progress.setValue(count)
                QtGui.qApp.processEvents()
                return progress.wasCanceled()

            try:
                sequence = self.queue.load_files(filelist,update)
            finally:
                progress.setValue(len(filelist))
                progress.hide()

            self.queue.set_frame_sequence(self.queue.get_frame_sequence() + sequence)
            self.frameListModel.updateFromQueue()

    def newFrame(self):
        pass

    def setStreaming(self, streaming):
        pass

    def setCount(self, count):
        pass

    def saveSequence(self):
        """
        Save sequence data to a file
        """
        path = self.settings.getString('input/file/default_path')
        filepath = QtGui.QFileDialog.getSaveFileName(self,
                                                     'Please choose a file where image sequence information will be saved',
                                                     path,
                                                     'Text files (*.txt);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.settings.setValue('input/file/default_path',new_path)
            f=open(filepath,'w')
            # write out the framerate
            f.write('framerate %lg\n' % self.frameRate.value())
            # write out each frame
            frames = self.queue.get_frame_sequence()
            for (framename, framepath, frameindex) in frames:
                f.write(repr(framename)+","+repr(framepath)+","+repr(frameindex)+"\n")
            f.close()
            QtGui.QMessageBox.information(self,
                                          'Image sequence saved',
                                          'Image sequence information has been saved to %s' % filepath)

    def loadSequence(self, dummy=False, parent=None):
        """
        Load sequence data from a file
        """
        if parent is None:
            parent = self
        path = self.settings.getString('input/file/default_path')
        filepath = QtGui.QFileDialog.getOpenFileName(parent,
                                                     'Please choose a file where image sequence information will be loaded',
                                                     path,
                                                     'Text files (*.txt);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.settings.setValue('input/file/default_path',new_path)
            try:
                f=open(filepath,'r')
                # read the lines
                lines=f.readlines()
                f.close()
                # read the framerate
                assert(lines[0].startswith('framerate '))
                framerate = float(lines[0][10:])
                sequence = []
                image_paths = {}
                # read each frame
                for line in lines[1:]:
                    if not line:
                        continue
                    framename, framepath, frameindex = eval(line, {"__builtins__":None}, {})
                    sequence.append((framename, framepath, frameindex))
                    image_paths[framepath] = True
            except Exception, e:
                QtGui.QMessageBox.critical(parent,
                                           'Error',
                                           'Unable to parse image sequence file')
                raise Error('Unable to parse image sequence file')

            filelist = sorted(image_paths.keys())
            progress = QtGui.QProgressDialog('Loading image sequence...',
                                             'Cancel',
                                             0,
                                             len(filelist),
                                             parent)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.show()
            progress.setValue(0)
            QtGui.qApp.processEvents()

            def update(count):
                progress.setValue(count)
                QtGui.qApp.processEvents()
                return progress.wasCanceled()

            try:
                # load the image data
                self.queue.load_files(filelist,update)
            finally:
                progress.setValue(len(filelist))
                progress.hide()

            # did we cancel?
            if progress.wasCanceled():
                self.queue.trim() # remove extra data
            else:
                # if we didn't, set the frame sequence and trim
                self.queue.set_frame_sequence(sequence)
                self.queue.trim()
                self.frameListModel.updateFromQueue()
                self.frameListWidget.selectionModel().clearSelection()

    def startup(self, parent=None):
        """
        The first action to be taken after widget is initialized
        """
        if parent is None:
            parent = self
        startup = self.settings.getInteger('input/file/multistartup',0)
        # 1: select images, 2: load from file sequence
        # 3: start with blank sequence, else: ask
        if startup not in [1,2,3]:
            # pop up a dialog
            dialog = gui.ChoiceDialog('What would you like to do?',
                                      ['Create a new sequence from images...',
                                       'Load a previously saved sequence...',
                                       'Create a new blank sequence'],
                                      parent=parent)
            value=dialog.exec_()
            startup = value
            if dialog.remember.isChecked():
                self.settings.setValue('input/file/multistartup',startup)
                self.startupButtons[startup].click()
        if startup == 1:
            self.addFiles(parent)
        elif startup == 2:
            self.loadSequence(parent=parent)
        elif startup == 3:
            pass



class MultipleFileInputPlugin:
    "Input plugin to load a file sequence"

    name = "Image file sequence..."
    description = "Load a sequence of image files as a light field movie."

    def __init__(self):
        pass

    def get_input(self, parent):
        """
        Display an input selection dialog if necessary and return an
        OpenedInput object

        Any exceptions raised would cause an error box to pop up
        """
        queue = MultipleFileQueue()
        widget = MultipleFileSettings(queue, parent.settings)
        widget.startup(parent)
        return inputs.OpenedInput(queue, widget)
        
    def close_input(self, opened_input):
        """
        Close an input that was previously opened by get_input

        The queue and widget correspond to the return values of the
        get_input() method
        """
        # clear the file list in the input 
        opened_input.queue().clear()
        del opened_input
