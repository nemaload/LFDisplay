"""
Output plugin management
"""

import Queue

from PyQt4 import QtCore

class OpenedOutput(QtCore.QObject):
    """
    A class for an open output

    One of these is created for an open output

    emitted signals:
      frameDone() <- indicates that the output is done with a frame
      recordingChanged(bool) <- indicates whether recording is on or off
      countDone(bool) <- indicates that we have recorded enough frames
      countChanged(int) <- indicates the counter has changed
      recordedFrameDone() <- indicates that a frame has been processed as well

    accepted slots:
      setRecording(bool) <- tell the output whether to start or stop recording
      setCount(int) <- tell the output that how many frames should be captured
                       counter is decremented by one each time a frame is
                       ready
      newFrame() <- tell the output that a new frame is ready
      startRecording() <- shorthand for setRecording(True)
      stopRecording() <- shorthand for setRecording(False)

    functions to override:
      processFrame() <- called by newFrame when a new frame has arrived
    """
    def __init__(self, widget, oneShotEnable=True):
        """
        Instantiate a newly opened output
        """
        QtCore.QObject.__init__(self)
        # our output queue
        self._queue = Queue.Queue(0)
        # for counting
        self._lock = QtCore.QMutex()
        self._count = 0
        # the controlling widget
        self._widget = widget
        # our input queue
        self._inputQueue = None
        # whether we are primed
        self._recording = False
        if oneShotEnable:
            # automatically shut off streaming when counter hits zero
            self.connect(self, QtCore.SIGNAL('countDone()'),
                         self.stopRecording)
        if self._widget:
            # allow the widget to know whether recording was stopped
            self.connect(self, QtCore.SIGNAL('recordingChanged(bool)'),
                         self._widget.setRecording)
            # allow the widget to know whether the count was changed
            self.connect(self, QtCore.SIGNAL('countChanged(int)'),
                         self._widget.setCount)
            # allow the widget to access the input
            self.connect(self._widget, QtCore.SIGNAL('recordingChanged(bool)'),
                         self.setRecording)
            self.connect(self._widget, QtCore.SIGNAL('countChanged(int)'),
                         self.setCount)
        
    def setInput(self, queue):
        """
        Set the input queue
        """
        self._lock.lock()
        self._inputQueue = queue
        self._lock.unlock()

    def setRecording(self, recording):
        """
        Set whether we are recording or not
        """
        self._lock.lock()
        recordingChanged = (self._recording != recording)
        self._recording = recording
        self._lock.unlock()
        if recordingChanged:
            self.emit(QtCore.SIGNAL('recordingChanged(bool)'),recording)

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

    def startRecording(self):
        self.setRecording(True)

    def stopRecording(self):
        self.setRecording(False)

    def recording(self):
        "Return whether we are recording or not"
        self._lock.lock()
        recording = self._recording
        self._lock.unlock()
        return recording

    def queue(self):
        """
        Return the *output* queue
        """
        return self._queue

    def widget(self):
        """
        Return the controlling widget
        """
        return self._widget

    def newFrame(self):
        """
        Handle a new incoming frame
        """
        countZero = False
        countDown = False
        countNum = 0
        self._lock.lock()
        recording = self._recording
        if recording:
            if self._count > 0:
                self._count -= 1
                countDown = True
                countNum = self._count
                if self._count == 0:
                    countZero = True
        inputQueue = self._inputQueue
        self._lock.unlock()
        try:
            # get the new frame
            frame = inputQueue.get(False)
            # process it
            if recording:
                self.processFrame(frame)
            # pass it on
            self._queue.put(frame)
        except Queue.Empty:
            print 'Warning, no frame'
            pass
        if countDown:
            self.emit(QtCore.SIGNAL('countChanged(int)'),countNum)
        if countZero:
            self.emit(QtCore.SIGNAL('countDone()'))
        self.emit(QtCore.SIGNAL('frameDone()'))
        if recording:
            self.emit(QtCore.SIGNAL('recordedFrameDone()'))

    def processFrame(self, frame):
        """
        Handle a new frame, override this for useful behavior
        """
        pass

class NullOutputPlugin:
    "A sample class for output plugins"

    # a descriptive name for the input
    name = "Null output"
    # a short description of this input
    description = "A dummy output for testing purposes only."

    def __init__(self):
        """
        Initialize the output and reset the output description if necessary.
        Throw an error if this output is not available
        """
        pass
    
    def get_output(self, parent):
        """
        Display an output selection dialog if necessary and return an
        OpenedOutput object

        Any exceptions raised would cause an error box to pop up
        """
        return OpenedOutput(None)

    def close_output(self, opened_output):
        """
        Close an output that was previously opened by get_output
        """
        pass

class OutputManager:
    "A class for managing output plugins"

    def __init__(self):
        self.clear()

    def clear(self):
        self.outputs = []
        self.output_classes = []

    def add_output(self, output_class):
        """
        Attempt to initialize an output class.
        If successful, the output is added to the output list.
        Otherwise, it is ignored.
        Only one output of each class is allowed

        Returns True if an output was added
        """
        if output_class in self.output_classes:
            return False # output already added
        try:
            output_instance = output_class()
            self.outputs.append(output_instance)
            self.output_classes.append(output_class)
            return True
        except Exception:
            return False

    def add_module(self, module):
        """
        Attempt to add all the output classes in a module automatically.
        Output class names must end in Output and must have a name string
        and a description string and must have a getOutput function

        Return the number of successfully added output classes.
        """
        possible_outputs = []
        for output_class in [module.__dict__[x] for x in module.__dict__ if x.endswith('OutputPlugin')]:
            try:
                # check for strings of name and description
                output_class.name + ''
                output_class.description + ''
                # check for call-ability of get_output
                if '__call__' not in dir(output_class.get_output):
                    continue
                possible_outputs.append(output_class)
            except Exception, e:
                continue
        # now attempt to add these outputs
        added_outputs = 0
        for output_class in possible_outputs:
            if self.add_output(output_class):
                added_outputs += 1
        # return the number of added outputs
        return added_outputs

    def get_outputs(self):
        """
        Return a copy of the list of outputs
        """
        return self.outputs[:]
