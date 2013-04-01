"""
Input plugin management
"""

import Queue

class Error(Exception):
    pass

class NullQueue:
    "A sample queue that returns no frames"

    # all queues must have an Empty exception as a member
    Empty = Queue.Empty
    
    def __init__(self):
        """
        Initialize the given input, etc.  Other queues may need extra
        arguments, like the camera number, etc.
        """
        pass

    def start(self):
        """
        Start streaming frames to the receivers
        """
        pass

    def stop(self):
        """
        Stop streaming frames to the receivers
        """
        pass

    def active(self):
        """
        Return whether frames are currently streaming
        """
        return False

    def pause(self):
        """
        Temporarily pause the streaming of frames
        """
        pass

    def unpause(self):
        """
        Undo a pause
        """
        pass

    def paused(self):
        """
        Return whether we are currently paused
        """
        return False

    def __del__(self):
        """
        The destructor should attempt to stop the queue
        """
        self.stop()

    def put(self, frame=None):
        """
        Add a frame, ready for capture, into the queue
        If None, a frame will be created.

        Normally, this should be done after the frame has
        been used by the receiver and this call indicates
        that the frame is ready for reuse
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
        raise self.Empty()

    def frame_done(self):
        """
        This function is called when a frame is ready on
        the queue.  Normally this is set to a different
        function instead of this dummy function by the receiver
        so that the receiver can get notifications
        """
        pass

from PyQt4 import QtCore

class OpenedInput(QtCore.QObject):
    """
    A class for an open input

    One of these is created for an open input

    emitted signals:
      frameDone() <- indicates that a frame is ready from the input
      streamingChanged(bool) <- indicates whether streaming is on or off
      pausedChanged(bool) <- indicates whether the input stream has been paused
                            (pausing/unpausing is quicker than
                            streaming/stopping)
      countDone(bool) <- indicates that we have captured enough frames
      countChanged(int) <- indicates the counter has changed
      desiredIntensityChanged(float) <- tells the display device that the requested
                                        apparent intensity of the input has changed
                                        differences between this intensity
                                        and the actual frame intensity should be
                                        applied to a display gain

    accepted slots:
      setStreaming(bool) <- tell the input whether to start or stop streaming
      setCount(int) <- tell the input that how many frames should be captured
                       counter is decremented by one each time a frame is
                       ready
      startStreaming() <- shorthand for setStreaming(True)
      stopStreaming() <- shorthand for setStreaming(False)
      setPaused(bool) <- tell the input whether to pause or continue streaming
                         pausing/unpausing should be quicker than
                         streaming/stopping

    methods:
      queue() <- return the associated queue
      widget() <- return the associated settings widget
      paused() <- returns whether we're paused or not
      count() <- returns the current count
      streaming() <- returns whether we're streaming or not
    """
    def __init__(self, queue, widget, oneShotEnable=True):
        """
        Instantiate a newly opened input

        queue is the input queue
        widget is the control widget
        """
        QtCore.QObject.__init__(self)
        
        self._queue = queue
        self._widget = widget
        self._lock = QtCore.QMutex()
        self._count = 0
        self._queue.frame_done = self._frameDone
        if oneShotEnable:
            # automatically shut off streaming when counter hits zero
            self.connect(self, QtCore.SIGNAL('countDone()'),
                         self.stopStreaming)
        if self._widget:
            # allow the widget to know whether streaming was stopped
            self.connect(self, QtCore.SIGNAL('streamingChanged(bool)'),
                         self._widget.setStreaming)
            # allow the widget to know whether the count was changed
            self.connect(self, QtCore.SIGNAL('countChanged(int)'),
                         self._widget.setCount)
            # allow the widget to access the input
            self.connect(self._widget, QtCore.SIGNAL('streamingChanged(bool)'),
                         self.setStreaming)
            self.connect(self._widget, QtCore.SIGNAL('countChanged(int)'),
                         self.setCount)
            # allow the widget to know when a frame was done
            self.connect(self, QtCore.SIGNAL('frameDone()'),
                         self._widget.newFrame)
            # allow the widget to notify the input that the desired intensity has changed
            self.connect(self._widget, QtCore.SIGNAL('desiredIntensityChanged(float)'),
                         self.setDesiredIntensity)

    def _frameDone(self):
        """
        Emit the frameDone signal and also the counterFinished signal
        if we were counting down
        """
        countZero = False
        countDown = False
        countNum = 0
        self._lock.lock()
        if self._count > 0:
            self._count -= 1
            countDown = True
            countNum = self._count
            if self._count == 0:
                countZero = True
        self._lock.unlock()
        self.emit(QtCore.SIGNAL('frameDone()'))
        if countDown:
            self.emit(QtCore.SIGNAL('countChanged(int)'),countNum)
        if countZero:
            self.emit(QtCore.SIGNAL('countDone()'))

    def queue(self):
        """
        Return our current queue
        """
        return self._queue

    def widget(self):
        """
        Return our controlling widget
        """
        return self._widget

    def startStreaming(self):
        self.setStreaming(True)

    def stopStreaming(self):
        self.setStreaming(False)

    def setStreaming(self, streaming):
        """
        Change whether we are streaming
        """
        oldStreaming = self._queue.active()
        if oldStreaming != streaming:
            if streaming:
                self._queue.start()
            else:
                self._queue.stop()
            self.emit(QtCore.SIGNAL('streamingChanged(bool)'),self._queue.active())

    def streaming(self):
        "Return whether we are streaming or not"
        return self._queue.active()

    def setPaused(self, paused):
        """
        Change whether we are paused
        """
        oldPaused = self._queue.paused()
        if oldPaused != paused:
            if paused:
                self._queue.pause()
            else:
                self._queue.unpause()
            self.emit(QtCore.SIGNAL('pausedChanged(bool)'),self._queue.paused())

    def paused(self):
        return self._queue.paused()

    def setCount(self, count):
        """
        Change how many frames should be captured.  Set to 0 for infinite
        """
        self._lock.lock()
        oldcount = self._count
        self._count = count
        self._lock.unlock()
        if oldcount != count:
            self.emit(QtCore.SIGNAL('countChanged(int)'),count)

    def count(self):
        "Return the current count"
        self._lock.lock()
        count = self._count
        self._lock.unlock()
        return count

    def setDesiredIntensity(self, desiredIntensity):
        self.emit(QtCore.SIGNAL('desiredIntensityChanged(float)'),desiredIntensity)
        

class NullInputPlugin:
    "A sample class for input plugins"

    # a descriptive name for the input
    name = "Null input"
    # a short description of this input
    description = "A dummy input for testing purposes only."

    def __init__(self):
        """
        Initialize the input and reset the input description if necessary.
        Throw an error if this input is not available
        """
        pass
    
    def get_input(self, parent):
        """
        Display an input selection dialog if necessary and return an
        OpenedInput object

        Any exceptions raised would cause an error box to pop up
        """
        return OpenedInput(NullQueue(), None)

    def close_input(self, opened_input):
        """
        Close an input that was previously opened by get_input
        """
        pass

class InputManager:
    "A class for managing input plugins"

    def __init__(self):
        self.clear()

    def clear(self):
        self.inputs = []
        self.input_classes = []

    def add_input(self, input_class):
        """
        Attempt to initialize an input class.
        If successful, the input is added to the input list.
        Otherwise, it is ignored.
        Only one input of each class is allowed

        Returns True if an input was added
        """
        if input_class in self.input_classes:
            return False # input already added
        try:
            input_instance = input_class()
            self.inputs.append(input_instance)
            self.input_classes.append(input_class)
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            return False

    def add_module(self, module):
        """
        Attempt to add all the input classes in a module automatically.
        Input class names must end in Input and must have a name string
        and a description string and must have a getInput function

        Return the number of successfully added input classes.
        """
        possible_inputs = []
        for input_class in [module.__dict__[x] for x in module.__dict__ if x.endswith('InputPlugin')]:
            try:
                # check for strings of name and description
                input_class.name + ''
                input_class.description + ''
                # check for call-ability of get_input
                if '__call__' not in dir(input_class.get_input):
                    continue
                possible_inputs.append(input_class)
            except Exception, e:
                continue
        # now attempt to add these inputs
        added_inputs = 0
        for input_class in possible_inputs:
            if self.add_input(input_class):
                added_inputs += 1
        # return the number of added inputs
        return added_inputs

    def get_inputs(self):
        """
        Return a copy of the list of inputs
        """
        return self.inputs[:]
