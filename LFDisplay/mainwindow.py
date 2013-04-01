"""
Main window for LFDisplay
"""

from PyQt4 import QtCore, QtGui

import os.path

import display

class SettingsPanel(QtGui.QDockWidget):
    """
    A basic settings panel widget
    """
    def __init__(self, name='', message='', widget=None, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.setObjectName(name)
        self.setWindowTitle(name)
        # the default label
        self.label = QtGui.QLabel(message)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        # the stack holding the label and setting page
        self.stack = QtGui.QStackedWidget()
        self.stack.addWidget(self.label)
        # the scroller holding the stack
        self.scroller = QtGui.QScrollArea()
        self.scroller.setWidget(self.stack)
        self.scroller.setWidgetResizable(True)
        # add the scoller
        self.setWidget(self.scroller)

        if widget:
            self.placeWidget(widget)

    def placeWidget(self, widget):
        "Place a widget into this setting panel and make it active"
        index = self.stack.addWidget(widget)
        self.stack.setCurrentIndex(index)

    def removeWidget(self, widget=None):
        "Remove a widget from the setting panel"
        if not widget: widget = self.stack.currentWidget()
        if widget == self.label: return
        self.stack.removeWidget(widget)

    def widget(self):
        return self.stack.currentWidget()

class SettingsPanelManager:
    """
    A manager for all the settings panels
    """
    def __init__(self, parent):
        self._parent = parent
        self._settings = []

    def add(self, panel):
        "Add a settings panel, initially all on the right in a tab"
        if panel not in self._settings:
            if self._settings:
                self._parent.tabifyDockWidget(self._settings[-1],panel)
            else:
                self._parent.addDockWidget(QtCore.Qt.RightDockWidgetArea,
                                           panel)
            self._settings.append(panel)
        else:
            raise Error('Attempting to add the same panel twice')

    def remove(self, panel):
        if panel in self._settings:
            self._parent.removeDockWidget(panel)
            self._settings.remove(panel)
        else:
            raise Error('Attempting to remove a panel that was not added')

    def __getitem__(self, key):
        for panel in self._settings:
            if panel.windowTitle() == key:
                return panel
        return None

    def toggleViewActions(self):
        "Get a list of view actions for all the settings panels"
        actions = [x.toggleViewAction() for x in self._settings]
        for x,y in zip(actions, self._settings):
            x.setText(y.windowTitle())
        return actions

class MainWindow(QtGui.QMainWindow):
    def __init__(self, settings, inputManager, outputManager, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        # set application settings
        self.settings = settings

        # set input stuff
        self.inputManager = inputManager
        self.closeInputFunc = None 
        self.openedInput = None
        self.streaming = False

        # set output stuff
        self.outputManager = outputManager
        self.closeOutputFunc = None
        self.openedOutput = None
        self.recording = False
        self.recordNum = 0

        # enumerate the inputs
        self.inputs = self.inputManager.get_inputs()

        # enumerate the outputs
        self.outputs = self.outputManager.get_outputs()
        
        # set display stuff
        self.zoom = 1.0

        # set our title
        self.setWindowIcon(QtGui.QIcon())
        self.setWindowTitle('Light Field Display')

        # create the display widget
        self.dispWidget = display.ImagingDisplay(self.settings, self)
        self.setCentralWidget(self.dispWidget)
        self.connect(self.dispWidget, QtCore.SIGNAL('zoomChanged(float)'),
                     self.changeZoom)
        # bind the destination of display widget to us
        self.connect(self.dispWidget,QtCore.SIGNAL('frameDone()'),self.retireFrame)
        self.retireQueue = self.dispWidget # grab from the display widget when done

        # set up the status bar
        self.statusBar_ = QtGui.QStatusBar(self)
        self.streamingStatus = QtGui.QLabel()
        self.streamingStatus.setMargin(2)
        self.recordingStatus = QtGui.QLabel()
        self.recordingStatus.setMargin(2)
        self.zoomStatus = QtGui.QLabel()
        self.zoomStatus.setMargin(2)
        self.recordNumStatus = QtGui.QLabel()
        self.recordNumStatus.setMargin(2)
        self.setStatus() # set a default status
        self.statusBar_.addWidget(self.streamingStatus)
        self.statusBar_.addWidget(self.recordingStatus)
        self.statusBar_.addWidget(self.zoomStatus)
        self.statusBar_.addWidget(self.recordNumStatus)
        self.setStatusBar(self.statusBar_)

        # setup our actions
        self.streamAction = QtGui.QAction(QtGui.QIcon(self.resource('play.png')),
                                          '&Stream',
                                          self)
        self.streamAction.setShortcut(QtCore.Qt.ALT + QtCore.Qt.Key_S)
        self.streamAction.setToolTip('Start/stop streaming frames from the camera.')
        self.streamAction.setCheckable(True)
        self.streamAction.setEnabled(False)
        self.connect(self.streamAction,
                     QtCore.SIGNAL('triggered(bool)'),
                     self.playTriggered)

        # make a pause icon
        self.pauseAction = QtGui.QAction(QtGui.QIcon(self.resource('pause.png')),
                                          '&Pause',
                                          self)
        self.pauseAction.setShortcut(QtCore.Qt.ALT + QtCore.Qt.Key_P)
        self.pauseAction.setToolTip('Pause/resume streaming frames from the camera.')
        self.pauseAction.setCheckable(True)
        self.pauseAction.setEnabled(False)
        self.connect(self.pauseAction,
                     QtCore.SIGNAL('triggered(bool)'),
                     self.pauseTriggered)

        self.recordAction = QtGui.QAction(QtGui.QIcon(self.resource('record.png')),
                                          '&Record',
                                          self)
        self.recordAction.setShortcut(QtCore.Qt.ALT + QtCore.Qt.Key_R)
        self.recordAction.setToolTip('Record the streamed frames to disk.')
        self.recordAction.setCheckable(True)
        self.recordAction.setEnabled(False)
        self.connect(self.recordAction,
                     QtCore.SIGNAL('triggered(bool)'),
                     self.record)

        self.quitAction = QtGui.QAction('&Quit', self)
        self.quitAction.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.quitAction.setToolTip('Exit the program.')
        self.connect(self.quitAction,
                     QtCore.SIGNAL('triggered(bool)'),
                     self.close)

        # add radio buttons
        self.displayPrefix = QtGui.QLabel(' View as: ')
        self.displayPrefix.setAlignment(QtCore.Qt.AlignVCenter)
        self.displayRawButton = QtGui.QRadioButton('Raw image')
        self.displayRawButton.setToolTip('Show the raw image, either coming from\nthe camera sensor or loaded from a file.')
        self.displayPinholeButton = QtGui.QRadioButton('Pinhole 3D (reset focus and pan)')
        self.displayPinholeButton.setToolTip('Show the light field as viewed through a pinhole.\nThis resets the focus and pan settings in the Optics tab.') 
        self.displayApertureButton = QtGui.QRadioButton('3D')
        self.displayApertureButton.setToolTip('Show a rendered 3D light field.')

        self.connect(self.displayRawButton,
                     QtCore.SIGNAL('clicked()'),
                     self.emitDisplayModeChanged)
        self.connect(self.displayPinholeButton,
                     QtCore.SIGNAL('clicked()'),
                     self.emitDisplayModeChanged)
        self.connect(self.displayApertureButton,
                     QtCore.SIGNAL('clicked()'),
                     self.emitDisplayModeChanged)

        # set up the toolbars
        self.controlBar = QtGui.QToolBar(self)
        self.controlBar.setObjectName('Control Bar')
        self.controlBar.addAction(self.streamAction)
        self.controlBar.addAction(self.pauseAction)
        self.controlBar.addAction(self.recordAction)
        self.displayBar = QtGui.QToolBar(self)
        self.displayBar.setObjectName('Display Mode Bar')
        self.displayBar.addWidget(self.displayPrefix)
        self.displayBar.addWidget(self.displayRawButton)
        self.displayBar.addWidget(self.displayPinholeButton)
        self.displayBar.addWidget(self.displayApertureButton)
        self.addToolBar(self.controlBar)
        self.addToolBar(self.displayBar)
        # toolbar view control
        self.viewControlBarAction = self.controlBar.toggleViewAction()
        self.viewControlBarAction.setText('&Controls')
        self.viewDisplayBarAction = self.displayBar.toggleViewAction()
        self.viewDisplayBarAction.setText('&Display mode')

        # set up the settings panels
        self.settingsManager = SettingsPanelManager(self)
        self.settingsManager.add(SettingsPanel(name = "Input",
                                               message = "No input selected.\nPlease open an input from the Data menu"
                                               ))
        self.settingsManager.add(SettingsPanel(name = "Output",
                                               message = "No output selected.\nPlease choose an output from the Data menu"
                                               ))
        self.settingsManager.add(SettingsPanel(name = "Display",
                                               message = "",
                                               widget = display.DisplaySettings(self.dispWidget)
                                               ))
        self.settingsManager.add(SettingsPanel(name = "Lenslet",
                                               message = "",
                                               widget = display.LensletSettings(self.dispWidget)
                                               ))
        self.settingsManager.add(SettingsPanel(name = "Optics",
                                               message = "",
                                               widget = display.OpticsSettings(self.dispWidget)
                                               ))
        
        # create the open input menu
        self.inputMenu = QtGui.QMenu('&Open input')
        self.inputActions = []
        # add in each input one by one
        for inputInstance in self.inputs:
            inputAction = self.inputMenu.addAction(inputInstance.name)
            inputAction.setStatusTip(inputInstance.description)
            #inputAction.setCheckable(True)
            self.connect(inputAction,
                         QtCore.SIGNAL('triggered(bool)'),
                         self.openInputInstance)
            self.inputActions.append(inputAction)
        # create the close input action
        self.closeInputAction = QtGui.QAction('&Close input',
                                              self)
        self.closeInputAction.setStatusTip('Close the currently open input.')
        self.closeInputAction.setEnabled(False)
        self.connect(self.closeInputAction,
                     QtCore.SIGNAL('triggered(bool)'),
                     self.closeInput)

        # create the open output menu
        self.outputMenu = QtGui.QMenu('O&pen output')
        self.outputActions = []
        # add in each output one by one
        defaultOutput = None
        for outputInstance in self.outputs:
            outputAction = self.outputMenu.addAction(outputInstance.name)
            outputAction.setStatusTip(outputInstance.description)
            #outputAction.setCheckable(True)
            if not defaultOutput:
                defaultOutput = outputAction
            self.connect(outputAction,
                         QtCore.SIGNAL('triggered(bool)'),
                         self.openOutputInstance)
            self.outputActions.append(outputAction)
        # create the close output action
        self.closeOutputAction = QtGui.QAction('C&lose output',
                                              self)
        self.closeOutputAction.setStatusTip('Close the currently open output.')
        self.closeOutputAction.setEnabled(False)
        self.connect(self.closeOutputAction,
                     QtCore.SIGNAL('triggered(bool)'),
                     self.closeOutput)

        # set up the menu bar
        self.menuBar_ = QtGui.QMenuBar(self)
        self.controlMenu = self.menuBar_.addMenu('&Control')
        self.controlMenu.addAction(self.streamAction)
        self.controlMenu.addAction(self.pauseAction)
        self.controlMenu.addAction(self.recordAction)
        self.controlMenu.addSeparator()
        self.controlMenu.addAction(self.quitAction)
        self.dataMenu = self.menuBar_.addMenu('&Data')
        self.dataMenu.addMenu(self.inputMenu)
        self.dataMenu.addAction(self.closeInputAction)
        self.dataMenu.addSeparator()
        self.dataMenu.addMenu(self.outputMenu)
        self.dataMenu.addAction(self.closeOutputAction)
        self.viewMenu = self.menuBar_.addMenu('&View')
        self.viewMenu.addAction(self.viewControlBarAction)
        self.viewMenu.addAction(self.viewDisplayBarAction)
        self.viewMenu.addSeparator()
        for action in self.settingsManager.toggleViewActions():
            self.viewMenu.addAction(action)
        self.setMenuBar(self.menuBar_)

        # set a sensible default for window size
        self.move(QtCore.QPoint(40,80))
        self.resize(QtCore.QSize(720,480))

        # load window settings
        try:
            self.resize(self.settings.value('main_window/size',self.size()).toSize())
        except Exception:
            pass
        try:
            self.move(self.settings.value('main_window/position',self.pos()).toPoint())
        except Exception:
            pass

        # load the previous state of the docks and toolbars
        try:
            state = QtCore.QByteArray(self.settings.getString('main_window/state').decode('hex'))
            self.restoreState(state)
        except Exception:
            # ignore
            pass

        # open the default output
        if defaultOutput:
            defaultOutput.trigger()

        # connect up display mode changes
        self.connect(self,
                     QtCore.SIGNAL('displayModeChanged(int)'),
                     self.dispWidget.processDisplayModeChanged)
        self.connect(self,
                     QtCore.SIGNAL('displayModeChanged(int)'),
                     self.settingsManager['Optics'].widget().processDisplayModeChanged)
        self.connect(self.settingsManager['Optics'].widget(),
                     QtCore.SIGNAL('displayModeChanged(int)'),
                     self.processDisplayModeChanged)
        self.connect(self.dispWidget,
                     QtCore.SIGNAL('displayModeChanged(int)'),
                     self.processDisplayModeChanged)

        # set the shader
        self.processDisplayModeChanged(self.dispWidget.nextShader())

    def processDisplayModeChanged(self, num):
        buttons = [self.displayRawButton,
                   self.displayPinholeButton,
                   self.displayApertureButton]
        if num >= 0 and num < len(buttons) and not buttons[num].isChecked():
            buttons[num].click()

    def emitDisplayModeChanged(self):
        """
        Emit a signal signaling that the display mode has changed
        """
        if self.displayRawButton.isChecked():
            newMode = 0
        elif self.displayPinholeButton.isChecked():
            newMode = 1
        elif self.displayApertureButton.isChecked():
            newMode = 2
        else:
            return
        self.emit(QtCore.SIGNAL('displayModeChanged(int)'), newMode)

    def resource(self, filename):
        """
        Return the actual location of a resource file
        """
        return os.path.join(self.settings.getString('app/resource_path'),filename)

    def openInputInstance(self):
        """
        Find out the sender of this signal and open the input
        """
        try:
            action = self.sender()
            instanceNumber = self.inputActions.index(action)
            instance = self.inputs[instanceNumber]
            openedInput = instance.get_input(self)
            if openedInput:
                # if user didn't cancel
                self.openInput(openedInput)
                self.closeInputFunc = instance.close_input
                # check the appropriate selection
                action.setChecked(True)
            else:
                action.setChecked(False)
        except Exception, e:
            QtGui.QMessageBox.critical(self, 'Error opening input', 'An error has occurred when trying to open an input:\n%s' % str(e))
            import traceback
            traceback.print_exc()

    def openInput(self, openedInput):
        """
        Open the input corresponding to the queue and settings widget
        """
        # close the old input, if necessary
        if self.openedInput:
            self.closeInput()
        # set our input
        self.openedInput = openedInput
        # add the settings widget
        widget = self.openedInput.widget()
        if widget:
            self.settingsManager['Input'].placeWidget(widget)
        else:
            self.settingsManager['Input'].placeWidget(QtGui.QLabel('The selected input has no settings.'))
        # bind the destination of the queue to display widget
        self.dispWidget.setQueue(self.openedInput.queue())
        self.connect(self.openedInput,QtCore.SIGNAL('frameDone()'),
                     self.dispWidget.newFrame)
        # bind some other stuff
        self.connect(self,QtCore.SIGNAL('streamingChanged(bool)'),
                     self.openedInput.setStreaming)
        self.connect(self.openedInput,QtCore.SIGNAL('streamingChanged(bool)'),
                     self.stream)
        self.connect(self.openedInput,QtCore.SIGNAL('desiredIntensityChanged(float)'),
                     self.dispWidget.setDesiredIntensity)
        # allow us to control stream
        self.streamAction.setEnabled(True)
        self.pauseAction.setEnabled(True)
        # allow us to close input
        self.closeInputAction.setEnabled(True)
        # start streaming
        self.stream(True)

    def closeInput(self):
        if not self.openedInput:
            return # no input was opened
        # don't allow us to close input
        self.closeInputAction.setEnabled(False)
        # don't allow us to stream
        self.streamAction.setEnabled(False)
        self.pauseAction.setEnabled(False)
        # stop streaming
        self.stream(False)
        # disconnect
        self.disconnect(self.openedInput,QtCore.SIGNAL('desiredIntensityChanged(float)'),
                     self.dispWidget.setDesiredIntensity)
        self.disconnect(self.openedInput,QtCore.SIGNAL('streamingChanged(bool)'),
                        self.stream)
        self.disconnect(self,QtCore.SIGNAL('streamingChanged(bool)'),
                        self.openedInput.setStreaming)
        self.disconnect(self.openedInput,QtCore.SIGNAL('frameDone()'),
                        self.dispWidget.newFrame)
        self.dispWidget.setQueue(None)
        self.settingsManager['Input'].removeWidget()
        # close the actual input
        if self.closeInputFunc:
            self.closeInputFunc(self.openedInput)
        self.openedInput = None
        # undo the checkmarks
        for inputAction in self.inputActions:
            inputAction.setChecked(False)

    def openOutputInstance(self):
        """
        Find out the sender of this signal and open the output
        """
        try:
            action = self.sender()
            instanceNumber = self.outputActions.index(action)
            instance = self.outputs[instanceNumber]
            openedOutput = instance.get_output(self)
            if openedOutput:
                # if user didn't cancel
                self.openOutput(openedOutput)
                self.closeOutputFunc = instance.close_output
                # check the appropriate selection
                action.setChecked(True)
            else:
                action.setChecked(False)
        except Exception, e:
            QtGui.QMessageBox.critical(self, 'Error opening output', 'An error has occurred when trying to open an output:\n%s' % str(e))
            import traceback
            traceback.print_exc()

    def openOutput(self, openedOutput):
        """
        Open the output corresponding to the queue and settings widget
        """
        # close the old output, if necessary
        if self.openedOutput:
            self.closeOutput()
        # set our output
        self.openedOutput = openedOutput
        # add the settings widget
        widget = self.openedOutput.widget()
        if widget:
            self.settingsManager['Output'].placeWidget(widget)
        else:
            self.settingsManager['Output'].placeWidget(LabeledWidget('Output','The selected output has no settings.'))
        # bind the output into the datapath
        self.openedOutput.setInput(self.dispWidget)
        self.disconnect(self.dispWidget,QtCore.SIGNAL('frameDone()'),self.retireFrame)
        self.retireQueue = self.openedOutput.queue()
        self.connect(self.openedOutput,QtCore.SIGNAL('frameDone()'),
                     self.retireFrame)
        self.connect(self.dispWidget,QtCore.SIGNAL('frameDone()'),
                     self.openedOutput.newFrame)
        self.connect(self.openedOutput,QtCore.SIGNAL('recordedFrameDone()'),
                     self.incrementRecordCounter)
        # bind output controls
        self.connect(self,QtCore.SIGNAL('recordingChanged(bool)'),
                     self.openedOutput.setRecording)
        self.connect(self.openedOutput,QtCore.SIGNAL('recordingChanged(bool)'),
                     self.record)
        # allow us to record
        self.recordAction.setEnabled(True)
        # allow us to close output
        self.closeOutputAction.setEnabled(True)

    def closeOutput(self):
        if not self.openedOutput:
            return # no output was opened
        # don't allow us to close output
        self.closeOutputAction.setEnabled(False)
        # don't allow us to record
        self.recordAction.setEnabled(False)
        # stop recordming
        self.record(False)
        # disconnect
        self.disconnect(self.openedOutput,QtCore.SIGNAL('recordedFrameDone()'),
                     self.incrementRecordCounter)
        self.disconnect(self.openedOutput,QtCore.SIGNAL('recordingChanged(bool)'),
                        self.record)
        self.disconnect(self,QtCore.SIGNAL('recordingChanged(bool)'),
                        self.openedOutput.setRecording)
        # disconnect datapath stuff
        self.disconnect(self.dispWidget,QtCore.SIGNAL('frameDone()'),
                        self.openedOutput.newFrame)
        self.disconnect(self.openedOutput,QtCore.SIGNAL('frameDone()'),
                        self.retireFrame)
        self.retireQueue = self.dispWidget
        self.connect(self.dispWidget,QtCore.SIGNAL('frameDone()'),
                     self.retireFrame)
        # remove the settings
        self.settingsManager['Output'].removeWidget()
        # close the actual output
        if self.closeOutputFunc:
            self.closeOutputFunc(self.openedOutput)
        self.openedOutput = None
        # undo the checkmarks
        for outputAction in self.outputActions:
            outputAction.setChecked(False)
        
    def retireFrame(self):
        """
        Process a frame and return a finished frame to the camera queue
        """
        # grab from last part in chain
        try:
            frame = self.retireQueue.get(False)
        except Exception:
            print 'Warning, no frame available'
            return
        
        # put into input queue
        if self.openedInput:
            self.openedInput.queue().put(frame)

    def incrementRecordCounter(self):
        "Increase the recorded frame number"
        self.setStatus(recordNum = self.recordNum + 1)

    def setStatus(self, streaming=None, recording=None, zoom=None, recordNum=None):
        """
        Handle the current status of the program
        """

        lastStreaming = self.streaming
        lastRecording = self.recording

        if None == streaming:
            streaming = self.streaming
        else:
            self.streaming = streaming
        if None == recording:
            recording = self.recording
        else:
            self.recording = recording
        if None == zoom:
            zoom = self.zoom
        else:
            self.zoom = zoom
        if None == recordNum:
            recordNum = self.recordNum
        else:
            self.recordNum = recordNum

        if streaming:
            self.streamingStatus.setText('STREAMING')
        else:
            self.streamingStatus.setText('STOPPED')
        if recording:
            if recording and streaming:
                self.recordingStatus.setText('RECORDING')
            else:
                self.recordingStatus.setText('READY TO RECORD')
        else:
            self.recordingStatus.setText('NOT RECORDING')
        if zoom >= 1.0:
            self.zoomStatus.setText('Zoom: %dX' % int(zoom))
        else:
            self.zoomStatus.setText('Zoom: 1/%dX' % int(1.0/zoom))
        self.recordNumStatus.setText('Frames recorded: %d' % self.recordNum)

        if lastStreaming != self.streaming or lastRecording != self.recording:
            self.streamingOrRecordingChanged(self.streaming, self.recording)
            if lastStreaming != self.streaming:
                self.streamingChanged(self.streaming)
            if lastRecording != self.recording:
                self.recordingChanged(self.recording)

    def playTriggered(self, state):
        """
        Handle when the play action is triggered
        """
        self.pauseAction.setChecked(not state)
        self.stream(state)
        
    def pauseTriggered(self, state):
        """
        Handle when the pause action is triggered
        """
        self.streamAction.setChecked(not state)
        self.stream(not state)

    def stream(self, streaming):
        """
        Start/stop streaming images from the camera
        """
        if self.openedInput:
            self.setStatus(streaming=streaming)
        else:
            self.setStatus(streaming=False)

    def record(self, recording):
        """
        Set whether we are recording frames to disk
        """
        self.setStatus(recording=recording)

    def streamingChanged(self, streaming):
        """
        When streaming state is changed
        """
        self.streamAction.setChecked(streaming)
        self.pauseAction.setChecked(not streaming)
        self.emit(QtCore.SIGNAL('streamingChanged(bool)'),streaming)

    def recordingChanged(self, recording):
        """
        When recording state is changed
        """
        self.recordAction.setChecked(recording)
        self.emit(QtCore.SIGNAL('recordingChanged(bool)'),recording)

    def streamingOrRecordingChanged(self, streaming, recording):
        """
        When streaming state is changed
        """
        pass
        
    def changeZoom(self, newZoom):
        """
        When zoom level is changed
        """
        self.setStatus(zoom=newZoom)

    def closeEvent(self, event):
        """
        When main window is closed
        """
        # close the input
        self.closeInput()
        # save the state
        state = self.saveState()
        self.settings.setValue('main_window/state', str(state.data()).encode('hex'))
        # save window settings
        self.settings.setValue('main_window/position', self.pos())
        self.settings.setValue('main_window/size', self.size())
        # close the window
        event.accept()

    def keyPressEvent(self, event):
        """
        Handle some shortcut keys
        """
        if event.key() == QtCore.Qt.Key_Plus:
            print 'plus'
        if event.key() == QtCore.Qt.Key_Plus and event.modifiers() == QtCore.Qt.ControlModifier:
            self.dispWidget.changeZoom(1.0)
        elif event.key() == QtCore.Qt.Key_Equal and event.modifiers() == QtCore.Qt.ControlModifier:
            self.dispWidget.changeZoom(1.0)
        elif event.key() == QtCore.Qt.Key_Minus and event.modifiers() == QtCore.Qt.ControlModifier:
            self.dispWidget.changeZoom(-1.0)
