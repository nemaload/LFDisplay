"""
An input plugin for QImaging cameras
"""

from PyQt4 import QtCore, QtGui
import math
import os
import sys
import traceback

import inputs
import gui

class Error(Exception):
    pass

# The control panel

class CameraSettings(QtGui.QWidget):
    """
    A window that has the various camera-specific settings
    """
    def __init__(self, camera, queue, settings, parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        self.camera = camera
        self.queue = queue
        self.settings = settings

        self.title = gui.TitleWidget(label='',
                                     title='QImaging camera',
                                     description='Stream images from a QImaging camera.')

        self.captureFormatGroup = QtGui.QGroupBox('Image output format', self)
        self.captureMono8 = QtGui.QRadioButton('Monochrome 8-bit')
        self.captureMono16 = QtGui.QRadioButton('Monochrome 16-bit')
        self.captureFormatLayout = QtGui.QGridLayout(self.captureFormatGroup)
        self.captureFormatLayout.setSpacing(0)
        self.captureFormatLayout.addWidget(self.captureMono8,0,0)
        self.captureFormatLayout.addWidget(self.captureMono16,0,1)
        self.captureFormatGroup.setLayout(self.captureFormatLayout)

        exposures = gui.ExponentialMap(self.camera.settings.min.exposure*1e-6,
                                       self.camera.settings.max.exposure*1e-6)

        self.exposureGroup = gui.SliderWidget(exposures,
                                              gui.TimeDisplay(),
                                              0.25,
                                              'Exposure',
                                              steps=999,
                                              compact=True)

        gains = gui.ExponentialMap(self.camera.settings.min.normalizedGain*1e-6,
                                   self.camera.settings.max.normalizedGain*1e-6)
        self.gainGroup = gui.SliderWidget(gains,
                                          (float,lambda x:'%.3g'%x),
                                          1.0,
                                          'Gain',
                                          steps=999,
                                          compact=True)

        num_adc_steps = self.camera.settings.max.absoluteOffset-self.camera.settings.min.absoluteOffset+1
        self.offsetGroup = gui.SliderWidget(gui.LinearMap(-1.0,1.0),
                                            (float,lambda x:'%.3g'%x),
                                            0.0,
                                            'Offset',
                                            steps=num_adc_steps,
                                            compact=True)

        self.binningGroup = QtGui.QGroupBox('Binning', self)
        self.binning1 = QtGui.QRadioButton('1x1 (No binning)')
        self.binning2 = QtGui.QRadioButton('2x2')
        self.binning4 = QtGui.QRadioButton('4x4')
        self.binning8 = QtGui.QRadioButton('8x8')
        self.binningLayout = QtGui.QGridLayout(self.binningGroup)
        self.binningLayout.addWidget(self.binning1,0,0)
        self.binningLayout.addWidget(self.binning2,1,0)
        self.binningLayout.addWidget(self.binning4,0,1)
        self.binningLayout.addWidget(self.binning8,1,1)
        self.binningGroup.setLayout(self.binningLayout)

        self.detailsGroup = QtGui.QGroupBox('Current parameters', self)
        self.details = QtGui.QTextEdit('')
        self.details.setReadOnly(True)
        self.detailsLayout = QtGui.QGridLayout(self.detailsGroup)
        self.detailsLayout.addWidget(self.details)
        self.detailsGroup.setLayout(self.detailsLayout)

        self.settingsLayout = QtGui.QGridLayout(self)
        self.settingsLayout.addWidget(self.title,0,0)
        self.settingsLayout.addWidget(self.captureFormatGroup,1,0)
        self.settingsLayout.addWidget(self.exposureGroup,2,0)
        self.settingsLayout.addWidget(self.gainGroup,3,0)
        self.settingsLayout.addWidget(self.offsetGroup,4,0)
        self.settingsLayout.addWidget(self.binningGroup,5,0)
        self.settingsLayout.addWidget(self.detailsGroup,6,0)
        self.settingsLayout.setRowStretch(6,1)
        self.setLayout(self.settingsLayout)

        self.settingsChanged = False # when camera settings need to
                                     # be flushed to camera
        self.timerInterval = 750 # make sure settings at least get updated this fast
        self.timerId = self.startTimer(self.timerInterval)
        self.lock = QtCore.QMutex()

        self.connect(self.captureMono8,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.captureFormatChanged)
        self.connect(self.captureMono16,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.captureFormatChanged)

        self.connect(self.exposureGroup,
                     QtCore.SIGNAL('valueSlid()'),
                     self.exposureChanged)
        self.connect(self.exposureGroup,
                     QtCore.SIGNAL('valueEdited()'),
                     self.exposureChanged)

        self.connect(self.gainGroup,
                     QtCore.SIGNAL('valueSlid()'),
                     self.gainChanged)
        self.connect(self.gainGroup,
                     QtCore.SIGNAL('valueEdited()'),
                     self.gainChanged)

        self.connect(self.offsetGroup,
                     QtCore.SIGNAL('valueSlid()'),
                     self.offsetChanged)
        self.connect(self.offsetGroup,
                     QtCore.SIGNAL('valueEdited()'),
                     self.offsetChanged)


        self.connect(self.binning1,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.binningChanged)
        self.connect(self.binning2,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.binningChanged)
        self.connect(self.binning4,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.binningChanged)
        self.connect(self.binning8,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.binningChanged)

    def timerEvent(self, e):
        """
        Flush settings to the camera
        """
        self.lock.lock()
        settingsChanged = self.settingsChanged
        self.lock.unlock()
        if not settingsChanged:
            return
        wasStreaming = self.queue.active()
        if wasStreaming:
            self.queue.pause()
        try:
            self.camera.settings.Flush()
        except OSError, e:
            print >> sys.stderr, "Camera has stopped working, ignoring:"
            traceback.print_exc()
        if wasStreaming:
            self.queue.unpause()
        self.updateFromCamera()
        self.lock.lock()
        self.settingsChanged = False
        self.lock.unlock()

    def newFrame(self):
        "A new frame has been received by the camera"
        self.timerEvent(None)

    def loadSettings(self):
        """
        Load previous settings
        """
        if self.settings.contains('input/qimaging/capture_format'):
            captureFormat = self.settings.getString('input/qimaging/capture_format')
            if captureFormat == 'mono8':
                self.camera.settings.imageFormat = 'mono8'
            elif captureFormat == 'mono16':
                self.camera.settings.imageFormat = 'mono16'
        if self.settings.contains('input/qimaging/exposure_us'):
            exposureUs = self.settings.getInteger('input/qimaging/exposure_us')
            self.camera.settings.exposure = exposureUs
            self.exposureGroup.setValue(exposureUs * 1e-6)
        if self.settings.contains('input/qimaging/normalized_gain'):
            gain = self.settings.getInteger('input/qimaging/normalized_gain')
            self.gainGroup.setValue(gain * 1e-6)
            self.camera.settings.normalizedGain = gain
        if self.settings.contains('input/qimaging/absolute_offset'):
            offset = self.settings.getInteger('input/qimaging/absolute_offset')
            self.offsetGroup.setValue(1.0*offset/self.camera.settings.max.absoluteOffset)
            self.camera.settings.absoluteOffset = offset
        self.camera.settings.Flush()
        self.updateFromCamera()

    def setStreaming(self, streaming):
        """
        When streaming is started or stopped
        """
        # stick hooks for the play button here
        
        # example: runs the command open when streaming starts
        #          and runs the command close when streaming stops
        #import os
        #if streaming:
        #    os.system('open')
        #else:
        #    os.system('close')
        #pass

    def setCount(self, count):
        """
        When the countdown counter has changed
        """
        pass

    def binningChanged(self):
        """
        When the binning changes
        """
        if self.binning1.isChecked():
            self.camera.settings.binning = 1 
        elif self.binning2.isChecked():
            self.camera.settings.binning = 2 
        elif self.binning4.isChecked():
            self.camera.settings.binning = 4 
        elif self.binning8.isChecked():
            self.camera.settings.binning = 8 
        # also change the roi
        self.camera.settings.roiX = 0
        self.camera.settings.roiY = 0
        self.camera.settings.roiWidth = self.camera.info.ccdWidth/self.camera.settings.binning
        self.camera.settings.roiHeight = self.camera.info.ccdHeight/self.camera.settings.binning
        wasStreaming = self.queue.active()
        if wasStreaming:
            self.queue.stop()
        self.camera.settings.Flush()
        if wasStreaming:
            self.queue.start()
        self.settings.setValue('input/qimaging/binning',self.camera.settings.binning)
        self.updateFromCamera()

    def captureFormatChanged(self):
        """
        When the capture format changes
        """
        if self.captureMono8.isChecked():
            self.camera.settings.imageFormat = 'mono8'
        elif self.captureMono16.isChecked():
            self.camera.settings.imageFormat = 'mono16'
        wasStreaming = self.queue.active()
        if wasStreaming:
            self.queue.stop()
        self.camera.settings.Flush()
        if wasStreaming:
            self.queue.start()
        self.settings.setValue('input/qimaging/capture_format',self.camera.settings.imageFormat)
        self.updateFromCamera()

    def exposureChanged(self):
        """
        When the exposure slider is adjusted
        """
        newExposure = int(round(self.exposureGroup.value() * 1e6))
        newExposure = min(max(self.camera.settings.min.exposure,newExposure),
                          self.camera.settings.max.exposure)
        oldExposure = self.camera.settings.exposure
        if oldExposure != newExposure:
            self.camera.settings.exposure = newExposure
            self.settings.setValue('input/qimaging/exposure_us',self.camera.settings.exposure)
            # queue a settings change
            self.lock.lock()
            self.settingsChanged = True
            self.lock.unlock()
            self.updateFromCamera()

    def gainChanged(self):
        """
        When the gain slider is adjusted
        """
        newGain = int(round(self.gainGroup.value() * 1e6))
        newGain = min(max(self.camera.settings.min.normalizedGain,newGain),
                        self.camera.settings.max.normalizedGain)
        oldGain = self.camera.settings.normalizedGain
        if oldGain != newGain:
            self.camera.settings.normalizedGain = newGain
            self.settings.setValue('input/qimaging/normalized_gain',self.camera.settings.normalizedGain)
            # queue a settings change
            self.lock.lock()
            self.settingsChanged = True
            self.lock.unlock()
            self.updateFromCamera()

    def offsetChanged(self):
        """
        When the offset slider is adjusted
        """
        newOffset = int(round(self.offsetGroup.value()*self.camera.settings.max.absoluteOffset))
        newOffset = min(max(self.camera.settings.min.absoluteOffset,newOffset),
                        self.camera.settings.max.absoluteOffset)
        oldOffset = self.camera.settings.absoluteOffset
        if oldOffset != newOffset:
            self.camera.settings.absoluteOffset = newOffset
            self.settings.setValue('input/qimaging/absolute_offset',self.camera.settings.absoluteOffset)
            # queue a settings change
            self.lock.lock()
            self.settingsChanged = True
            self.lock.unlock()
            self.updateFromCamera()

    def updateFromCamera(self):
        """
        Update the values from the camera
        """
        if not self.camera:
            return
        # update the details
        details = []
        details.append(('Camera', self.camera.info.cameraType))
        details.append(('Sensor', ('%s (%dx%dx%d bit %gx%g um pixels)') % (self.camera.info.ccd[0],self.camera.info.ccdWidth,self.camera.info.ccdHeight,self.camera.info.bitDepth,self.camera.info.ccd[1],self.camera.info.ccd[2])))
        details.append(('Exposure', '%d us' % self.camera.settings.exposure))
        details.append(('Binning', '%dx%d' % (self.camera.settings.binning,
                                              self.camera.settings.binning)))
        details.append(('Gain', '%d.%06d' %
                        (int(self.camera.settings.normalizedGain/1000000),
                         self.camera.settings.normalizedGain % 1000000)))
        details.append(('Offset', '%d/%d' %
                        (self.camera.settings.absoluteOffset,
                         self.camera.settings.max.absoluteOffset)))
        details.append(('Image output format', self.camera.settings.imageFormat))
        detailsText = '<br>'.join([('<b>'+x+':</b> '+y) for (x,y) in details])
        self.details.setText(detailsText)
        # get the capture format
        if self.camera.settings.imageFormat == 'mono16':
            self.captureMono16.setChecked(True)
        elif self.camera.settings.imageFormat == 'mono8':
            self.captureMono8.setChecked(True)
        # get binning sections
        if self.camera.settings.binning == 1:
            self.binning1.setChecked(True)
        elif self.camera.settings.binning == 2:
            self.binning2.setChecked(True)
        elif self.camera.settings.binning == 4:
            self.binning4.setChecked(True)
        elif self.camera.settings.binning == 8:
            self.binning8.setChecked(True)
        # notify of intensity changes
        gainExposure = 0.000001 * self.camera.settings.exposure * 0.000001 * self.camera.settings.normalizedGain
        self.emit(QtCore.SIGNAL('desiredIntensityChanged(float)'),gainExposure)

# the input plugin

class QCamInputPlugin:
    "Input plugin for QImaging cameras"

    # a descriptive name for the input
    name = "QImaging camera"
    # a short description of this input
    description = "Capture images from a QImaging camera."

    def __init__(self):
        """
        Initialize the input and reset the input description if necessary.
        Throw an error if this input is not available
        """
        from qcam import QCam, QCamUtil
        # save our working directory
        cwd=os.getcwd()
        # attempt to load the driver
        QCam.LoadDriver()
        # load our working directory
        os.chdir(cwd)
        # try releasing it
        QCam.ReleaseDriver()
    
    def get_input(self, parent):
        """
        Display an input selection dialog if necessary and return
        an OpenedInput

        Any exceptions raised would cause an error box to pop up
        """
        from qcam import QCam, QCamUtil
        cwd=os.getcwd()
        QCam.ReleaseDriver()
        QCam.LoadDriver()
        os.chdir(cwd)
        
        # list the cameras the camera
        cameraList = QCam.ListCameras()
        if 0 >= len(cameraList):
            raise Error('No cameras found.')

        # open the camera
        camera = QCam.OpenCamera(cameraList[0])

        # set a default exposure, format, etc.
        camera.settings.exposure = 25000
        camera.settings.imageFormat = 'mono16'
        camera.settings.Flush()

        # set up camera queue
        queue = QCamUtil.CameraQueue(camera)

        # create a settings widget
        widget = CameraSettings(camera, queue, parent.settings)
        widget.loadSettings()
        widget.updateFromCamera()

        # return an opened input
        return inputs.OpenedInput(queue, widget)

    def close_input(self, opened_input):
        """
        Close an input that was previously opened by get_input
        """
        from qcam import QCam, QCamUtil
        try:
            # stop streaming
            opened_input.queue().stop()
        except Exception, e:
            print >> sys.stderr, "Ignored exception:"
            traceback.print_exc()
        try:
            # close the camera
            opened_input.queue().camera.CloseCamera()
        except Exception, e:
            print >> sys.stderr, "Ignored exception:"
            traceback.print_exc()
        try:
            # unload our drivers
            QCam.ReleaseDriver()
        except Exception, e:
            print >> sys.stderr, "Ignored exception:"
            traceback.print_exc()
