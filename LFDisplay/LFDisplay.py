#!/usr/bin/python
"""
Main application for LFDisplay
"""

from PyQt4 import QtCore, QtGui
import sys
import os
import os.path

# allow eggs to be dropped into application folder, as well as script
# overrides, etc.
sys.path = ['.'] + sys.path

from settings import Settings

# create the application
app = QtGui.QApplication(sys.argv)

# load up settings
qt_settings = QtCore.QSettings(QtCore.QSettings.IniFormat,
                               QtCore.QSettings.UserScope,
                               'Stanford University',
                               'LFDisplay')
settings = Settings(qt_settings)

# set defaults
if not settings.contains('output/default_path'):
    settings.setValue('output/default_path',os.getcwd())
if not settings.contains('input/default_path'):
    settings.setValue('input/default_path',os.getcwd())
if not settings.contains('app/resource_path') or not os.path.exists(os.path.join(settings.getString('app/resource_path'), 'splash.png')):
    # load resource location
    cwd = os.getcwd()
    if not sys.argv[0]:
        resource_path = cwd
    else:
        resource_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    settings.setValue('app/resource_path',resource_path)


# Show the splash screen
splash_path = os.path.join(settings.getString('app/resource_path'), 'splash.png')
splash = QtGui.QSplashScreen(QtGui.QPixmap(splash_path))
splash.show()

# set up my application
QtCore.QCoreApplication.setOrganizationName('Stanford University')
QtCore.QCoreApplication.setOrganizationDomain('stanford.edu')
QtCore.QCoreApplication.setApplicationName('LFDisplay')

# loading input plugins
splash.showMessage('Loading input plugins...', QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)

import inputs

inputManager = inputs.InputManager()

import qimaging
import fileinput

# add the qimaging input
inputManager.add_module(qimaging)
# add the file inputs
inputManager.add_module(fileinput)

# loading output plugins
splash.showMessage('Loading output plugins...', QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)

import outputs

outputManager = outputs.OutputManager()

import fileoutput

# add the file outputs
outputManager.add_module(fileoutput)

# load the main window
splash.showMessage('Loading main window...', QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)

import mainwindow

mainWindow = mainwindow.MainWindow(settings, inputManager, outputManager)

mainWindow.show()

splash.finish(mainWindow)

# run the application
result=app.exec_()

# exit
sys.exit(result)
