#!/usr/bin/env python

from distutils.core import setup

setup(name='QCam',
      version='1.0.1',
      description='Python bindings for the QCam SDK',
      author='Zhengyun Zhang',
      package_dir = {'qcam':''},
      py_modules=['qcam.QCam','qcam._qcam','qcam.QCamUtil','qcam.__init__']
      )
