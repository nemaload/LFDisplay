"""
Image file output plugins
"""

import time
import Queue
import thread
import os
import re

import numpy
import math
import struct
import os
from PyQt4 import QtCore, QtGui

import outputs
import gui

import _FreeImage

numDigits = 8

class Error(Exception):
    pass

class DummyFrame:
    pass

class ActionQueue(object):
    "A singleton object that handles an output file queue"
    def __new__(cls, *args, **kwds):
        instance = cls.__dict__.get('__instance__')
        if instance is not None:
            return instance
        instance = object.__new__(cls)
        cls.__instance__ = instance
        instance.init(*args, **kwds)
        return instance
    
    def __init__(self):
        "This gets called multiple times"
        pass

    def worker(self):
        "Worker thread"
        done = False
        while not done:
            job = self._queue.get()
            if not job:
                done = True
            else:
                func, params = job
                func(*params)
        # clear the queue when done
        self._queue.clear()
        
    def init(self, *args, **kwds):
        "This gets called only once"
        self._queue = Queue.Queue()
        
    def start(self):
        "Start the action queue"
        thread.start_new_thread(self.worker, ())

    def stop(self):
        "Stop the action queue"
        self._queue.put(None)

    def add(self, func, params):
        "Add an item to the queue"
        self._queue.put((func,params))

# start the queue
ActionQueue().start()

def _write_tmp(arr, filename, width, height, channels, logfile):
    print >> logfile, str(time.time())+' Saving '+filename+'...',

    try:
        # make the directory if it doesn't already exist
        path, fname = os.path.split(filename)
        if not os.path.isdir(path):
            os.makedirs(path)
            
        f=open(filename,'wb')
        f.write(struct.pack('iiii',1,width,height,channels))
        f.write(arr.tostring())
        f.close()
        print >> logfile, 'SUCCESS'
    except Exception, e:
        print >> logfile, 'FAILED: '+str(e)

def _write_fimg(dib, filename, logfile):
    print >> logfile, str(time.time())+' Saving '+filename+'...',
    
    try:
        # make the directory if it doesn't already exist
        path, fname = os.path.split(filename)
        if not os.path.isdir(path):
            os.makedirs(path)
        fif = _FreeImage.GetFIFFromFilename(filename)
        options = 0 # output options
        if fif == _FreeImage.FIF_TIFF:
            options = _FreeImage.TIFF_NONE # turn off compression
        _FreeImage.Save(fif, dib, filename, options)
        _FreeImage.Unload(dib)
        print >> logfile, 'SUCCESS'
    except Exception, e:
        print >> logfile, 'FAILED: '+str(e)
        import traceback
        traceback.print_exc()

def _save_frame(frame, filename, logfile, rescale=True):
    """
    Save a frame into an output image
    """
    max_bit_depth = frame.bits
    if frame.formatString == 'mono8':
        numpy_format = numpy.uint8
        fit = _FreeImage.FIT_BITMAP
        bit_depth = 8
        channels = 1
    elif frame.formatString == 'mono16':
        numpy_format = numpy.uint16
        fit = _FreeImage.FIT_UINT16
        bit_depth = 16
        channels = 1
    elif frame.formatString in ['rgb24','bgr24']:
        # TODO color ordering
        numpy_format = numpy.uint8
        fit = _FreeImage.FIT_BITMAP
        bit_depth = 8
        channels = 3
    elif frame.formatString == 'rgb48':
        numpy_format = numpy.uint16
        fit = _FreeImage.FIT_RGB16
        bit_depth = 16
        channels = 3
    else:
        raise Error('Unsupported frame format: '+frame.formatString)
    
    actual_bit_depth = min(bit_depth, max_bit_depth)
    if filename[-4:].lower() == '.tmp':
        # imagestack tmp file
        arr=numpy.fromstring(frame.stringBuffer, dtype=numpy_format, count=frame.width*frame.height*channels)
        arr=arr.astype('float32')
        if rescale:
            scale_factor = math.pow(2.0, -actual_bit_depth)
            arr = arr * scale_factor
        ActionQueue().add(_write_tmp, (arr, filename, frame.width, frame.height, channels, logfile))
    else:
        # normal image file
        if rescale and bit_depth != max_bit_depth:
            arr=numpy.fromstring(frame.stringBuffer, dtype=numpy_format, count=frame.width*frame.height*channels)
            scale_factor = 1 << (bit_depth - actual_bit_depth)
            arr = arr * scale_factor
            data = arr.tostring()
        else:
            data = frame.stringBuffer
        dib = _FreeImage.ConvertFromRawBitsT(data, fit,
                                             frame.width, frame.height,
                                             frame.width*bit_depth*channels/8,
                                             bit_depth*channels,
                                             topdown=False)
        if frame.formatString == 'mono8':
            _FreeImage.SetPaletteMono8(dib)
        # do stuff
        ActionQueue().add(_write_fimg, (dib,filename,logfile))

class ImagesOutputSettings(QtGui.QWidget):
    """
    Settings for image output
    """
    def __init__(self, settings, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.settings = settings # application level settings

        outputPath = self.settings.getString('output/default_path','.')
        outputPath = self.settings.getString('output/file/default_path',outputPath)
        sequenceNumbered = self.settings.getBool('output/file/sequence_numbered',False)

        sequenceName = self.settings.getString('output/file/sequence_name','capture')
        sequenceExtension = self.settings.getString('output/file/sequence_extension','tif')
        sequenceBurst = self.settings.getInteger('output/file/sequence_burst',1)
        sequenceBurstEnabled = self.settings.getBool('output/file/sequence_burst_enabled',False)

        # create the UI
        self.title = gui.TitleWidget(label='',
                                     title='Image file(s)',
                                     description='Save captured images to disk as individual files.')
        self.outputPathSelector = gui.PathSelectorWidget(label='Output folder',
                                                         browseCaption='Please choose a folder where new output image(s) will be created.',
                                                         default=outputPath)

        self.outputFileGroup = QtGui.QGroupBox('Output filename(s)')
        label1 = QtGui.QLabel('Name:')
        label1.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.outputFileHeader = QtGui.QLineEdit(sequenceName, self.outputFileGroup)
       
        label2 = QtGui.QLabel('Image type:')
        label2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        image_types = [('TIFF','tif'),
                       ('ImageStack TMP','tmp')]
        self.outputFileExtension = gui.CustomOptionSelectorWidget(caption='',
                                                                  options=image_types,
                                                                  custom=None,
                                                                  parent=self.outputFileGroup)

        label3 = QtGui.QLabel('Options:')
        label3.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.limitFrames = QtGui.QCheckBox('Limit frames per recording:')
        self.framesToRecord = QtGui.QLineEdit('1')
        validator = QtGui.QIntValidator(self.framesToRecord)
        validator.setBottom(1)
        self.framesToRecord.setValidator(validator)

        self.numFramesLayout = QtGui.QHBoxLayout()
        self.numFramesLayout.setMargin(0)
        self.numFramesLayout.addWidget(self.limitFrames)
        self.numFramesLayout.addWidget(self.framesToRecord)
        
        self.appendFrameNumber = QtGui.QCheckBox('Append frame number')

        self.label4 = QtGui.QLabel('Next frame:')
        self.label4.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.outputFileNumberLayout = QtGui.QHBoxLayout()
        self.outputFileNumberLayout.setMargin(0)

        self.outputFileNumber = QtGui.QLineEdit('0')
        validator = QtGui.QIntValidator(self.outputFileNumber)
        validator.setBottom(0)
        self.outputFileNumber.setValidator(validator)
        self.autoNumber = QtGui.QPushButton('Auto')
        self.outputFileNumberLayout.addWidget(self.outputFileNumber)
        self.outputFileNumberLayout.addWidget(self.autoNumber)


        self.outputFileLayout = QtGui.QGridLayout(self.outputFileGroup)
        self.outputFileLayout.addWidget(label1, 0, 0)
        self.outputFileLayout.addWidget(self.outputFileHeader, 0, 1)
        self.outputFileLayout.addWidget(label2, 1, 0)
        self.outputFileLayout.addWidget(self.outputFileExtension, 1, 1)
        self.outputFileLayout.addWidget(label3, 2, 0)
        self.outputFileLayout.addLayout(self.numFramesLayout, 2, 1)
        self.outputFileLayout.addWidget(self.appendFrameNumber, 3, 1)
        
        self.outputFileLayout.addWidget(self.label4, 50, 0)
        self.outputFileLayout.addLayout(self.outputFileNumberLayout, 50, 1)
        self.outputFileLayout.setRowStretch(99,1)
        self.outputFileGroup.setLayout(self.outputFileLayout)

        self.controlsGroup = QtGui.QGroupBox('Controls')

        self.snapshot = QtGui.QPushButton('Save current image')
        #self.startStream = QtGui.QPushButton('Start recording stream')
        #self.stopStream = QtGui.QPushButton('Stop recording stream')

        self.controlsLayout = QtGui.QGridLayout(self.controlsGroup)
        self.controlsLayout.addWidget(self.snapshot, 0, 0)
        #self.controlsLayout.addWidget(self.startStream, 0, 1)
        #self.controlsLayout.addWidget(self.stopStream, 0, 2)
        self.controlsLayout.setColumnStretch(99,1)
        self.controlsGroup.setLayout(self.controlsLayout)


        self.log = gui.LogWidget('Output log')


        self.settingsLayout = QtGui.QGridLayout(self)
        self.settingsLayout.addWidget(self.title,0,0)
        self.settingsLayout.addWidget(self.outputPathSelector,1,0)
        self.settingsLayout.addWidget(self.outputFileGroup,2,0)
        self.settingsLayout.addWidget(self.controlsGroup,3,0)
        self.settingsLayout.addWidget(self.log,4,0)
        self.settingsLayout.setRowStretch(99,1)
        self.setLayout(self.settingsLayout)

        # connect the signals
        self.connect(self.outputPathSelector, QtCore.SIGNAL('textChanged(QString)'),
                     self.setPath)
        
        self.connect(self.outputFileHeader, QtCore.SIGNAL('textChanged(QString)'),
                     self.createFilename)
        
        self.connect(self.outputFileExtension, QtCore.SIGNAL('valueChanged(QString)'),
                     self.createFilename)

        self.connect(self.limitFrames, QtCore.SIGNAL('stateChanged(int)'),
                     self.burstChanged)
        self.connect(self.framesToRecord, QtCore.SIGNAL('textChanged(QString)'),
                     self.burstChanged)
        
        self.connect(self.appendFrameNumber, QtCore.SIGNAL('stateChanged(int)'),
                     self.numberedChecked)
        self.connect(self.appendFrameNumber, QtCore.SIGNAL('stateChanged(int)'),
                     self.createFilename)

        self.connect(self.outputFileNumber, QtCore.SIGNAL('textChanged(QString)'),
                     self.checkNumber)
        self.connect(self.outputFileNumber, QtCore.SIGNAL('textChanged(QString)'),
                     self.setNumberFromField)
        
        self.connect(self.autoNumber, QtCore.SIGNAL('clicked()'),
                     self.setAutoNumber)

        # set the defaults
        self.limitFrames.setChecked(sequenceBurstEnabled)
        self.framesToRecord.setText(str(sequenceBurst))
        self.appendFrameNumber.setChecked(sequenceNumbered)
        self.numberedChecked()
        self.setAutoNumber()
        self.outputFileHeader.setText(sequenceName)
        try:
            self.outputFileExtension.setValue(str(sequenceExtension))
        except Exception:
            pass

    def burstChanged(self):
        self.settings.setValue('output/file/sequence_burst', int(self.framesToRecord.text()))
        self.settings.setValue('output/file/sequence_burst_enabled', bool(self.limitFrames.isChecked()))

    def setRecording(self, recording):
        if recording:
            if self.settings.getBool('output/file/sequence_burst_enabled', False):
                self.emit(QtCore.SIGNAL('countChanged(int)'), self.settings.getInteger('output/file/sequence_burst'))
            else:
                self.emit(QtCore.SIGNAL('countChanged(int)'), 0)

    def setCount(self, count):
        pass

    def setNumberFromField(self):
        number = self.outputFileNumber.text()
        if number:
            self.emit(QtCore.SIGNAL("numberChanged(int)"), int(number))

    def setNumber(self, number):
        self.outputFileNumber.setText(str(number))

    def setPath(self, s):
        if s != self.outputPathSelector.text():
            self.outputPathSelector.setText(s)
        self.settings.setValue('output/file/default_path',s)

    def numberedChecked(self):
        numbered = self.appendFrameNumber.isChecked()
        self.settings.setValue('output/file/sequence_numbered', numbered)
        self.label4.setEnabled(numbered)
        self.outputFileNumber.setEnabled(numbered)
        self.autoNumber.setEnabled(numbered)
        self.setAutoNumber()

    def createFilename(self):
        # read the filename info
        header = str(self.outputFileHeader.text())
        numbered = self.appendFrameNumber.isChecked()
        extension = str(self.outputFileExtension.value())
        self.settings.setValue('output/file/sequence_name', header)
        self.settings.setValue('output/file/sequence_extension', extension)
        self.settings.setValue('output/file/sequence_numbered', numbered)
        # compose the new filename
        if numbered:
            filename = header+('-%%0%dd.' % numDigits)+extension
        else:
            filename = header+'.'+extension
        self.setFilename(filename)
        self.checkNumber()

    def checkNumber(self):
        numbered = self.appendFrameNumber.isChecked()
        paletteGood = QtGui.QPalette()
        paletteBad = QtGui.QPalette()
        paletteBad.setColor(QtGui.QPalette.Base,
                            QtGui.QColor("Red"))
        maxNumber = self.getMaxNumber()
        if numbered:
            self.outputFileHeader.setPalette(paletteGood)
            try:
                number = int(self.outputFileNumber.text())
                if maxNumber >= number:
                    # make it red if it's going to overwrite
                    self.outputFileNumber.setPalette(paletteBad)
                else:
                    self.outputFileNumber.setPalette(paletteGood)
            except ValueError:
                self.outputFileNumber.setPalette(paletteBad)
        else:
            # check to make sure file exists
            if maxNumber == 0:
                # file exists
                self.outputFileHeader.setPalette(paletteBad)
            else:
                self.outputFileHeader.setPalette(paletteGood)

    def getMaxNumber(self):
        "Find the highest numbered file that currently exists or -1 if none"
        header = self.settings.getString('output/file/sequence_name','capture')
        extension = self.settings.getString('output/file/sequence_extension','tif')
        numbered = self.settings.getBool('output/file/sequence_numbered',False)
        filepath = self.settings.getString('output/file/default_path','.')
        try:
            files = [x.lower() for x in os.listdir(filepath)]
        except Exception, e:
            # could not list directory
            return -1
        if not numbered:
            if (header+'.'+extension).lower() in files:
                return 0
            else:
                return -1
        # construct regular expression for looking for files
        def escape(s):
            if s in ".^$*+?{}[]\\()|":
                return "\\"+s
            else:
                return s
        header_re = "^"+''.join([escape(x.lower()) for x in header])
        middle_re = "-"+"("+"[0-9]"*numDigits+")"
        ext_re = r"\." + ''.join([escape(x.lower()) for x in extension])+ "$"

        full_re = header_re+middle_re+ext_re
        full_re2 = re.compile(full_re)
        # find all the file matches
        numbers = [int(x.groups()[0]) for x in [full_re2.match(y) for y in files] if x]
        if numbers:
            return max(numbers)
        else:
            return -1

    def setAutoNumber(self):
        "Set the number automatically"
        self.outputFileNumber.setText(str(self.getMaxNumber()+1))

    def setFilename(self, s):
        self.settings.setValue('output/file/output_filename',s)

class ImagesOpenedOutput(outputs.OpenedOutput):
    def __init__(self, settings):
        self.settingsPanel = ImagesOutputSettings(settings)
        outputs.OpenedOutput.__init__(self, self.settingsPanel)

        self.number = 0
        self.settings = settings

        self.connect(self, QtCore.SIGNAL('numberChanged(int)'),
                     self.settingsPanel.setNumber)
        self.connect(self.settingsPanel, QtCore.SIGNAL('numberChanged(int)'),
                     self.setNumber)
        self.connect(self.settingsPanel.snapshot, QtCore.SIGNAL('clicked()'),
                     self.grabCurrentFrame)
            

    def processFrame(self, frame):
        outputFile = self.settings.getString('output/file/output_filename','output.tif')
        try:
            filename = outputFile % (self.number)
        except TypeError:
            filename = outputFile


        outputPath = self.settings.getString('output/default_path')
        outputPath = self.settings.getString('output/file/default_path',outputPath)
        filename = os.path.join(outputPath, filename)

        self.setOutputPath(outputPath)

        try:
            _save_frame(frame, filename, self.settingsPanel.log)
        except Exception, e:
            print >> self.settingsPanel.log, "Unable to save frame: " + str(e)
            import traceback
            traceback.print_exc()

        self.setNumber(self.number+1)

    def grabCurrentFrame(self):
        "Grab the current frame from the display widget"
        frame = DummyFrame()
        display = self._inputQueue # this should be the display widget
        display.lock.lock()
        if not display.currentTexture:
            display.lock.unlock()
            return
        frame.width, frame.height = display.newTextureSize
        frame.stringBuffer = display.currentTexture[:]
        frame.formatString = display.textureFormat
        frame.bits = display.textureBits
        frame.intensity = display.newFrameIntensity
        display.lock.unlock()
        self.processFrame(frame)

    def setNumber(self, number):
        "A slot to change the current frame number"
        changed = self.number != number
        self.number = number
        if changed:
            self.emit(QtCore.SIGNAL('numberChanged(int)'),number)

    def setOutputPath(self, path):
        "A slot to change the output path"
        self.settings.setValue('output/file/default_path',path)

    def setOutputFileName(self, outputFileName):
        changed = self.outputFileName != outputFileName
        self.outputFileName = outputFileName
        if changed:
            self.emit(QtCore.SIGNAL('outputFileNameChanged(QString)'),outputFileName)

class ImagesOutputPlugin:
    "An output plugin to capture images" 

    # a descriptive name for the input
    name = "Image file(s)"
    # a short description of this input
    description = "Record data to image(s) on disk."

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
        return ImagesOpenedOutput(parent.settings)

    def close_output(self, opened_output):
        """
        Close an output that was previously opened by get_output
        """
        pass
