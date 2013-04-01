"""
A build script for packaging LFDisplay_FreeImage
"""

import platform
import sys

class Error(Exception):
    pass

app_title = 'LFDisplay_FreeImage'

if platform.system() == 'Windows':
    from distutils.core import setup
    import py2exe
    import ctypes.util
    import os.path
    import glob
    import OpenGL
    import shutil
    import zipfile
    import os
    import time

    dist_name = 'LFDisplay-Windows-'+platform.machine()+'-'+time.strftime('%Y%m%d',time.localtime(time.time()))
    print 'Packaging '+dist_name

    # determine if SxS libraries for the Python exe need to be copied over
    exe_path = os.path.dirname(sys.executable)
    msvcr_dlls = glob.glob(os.path.join(exe_path, "MSVCR*.dll"))
    manifests = glob.glob(os.path.join(exe_path, "Microsoft.VC*.CRT.manifest"))
    print "Found Microsoft runtime DLLs:"
    print "\n".join(msvcr_dlls)
    print "Found Microsoft SxS manifests:"
    print "\n".join(manifests)

    # find where FreeImage is so that we can copy it
    FreeImage_location = ctypes.util.find_library('FreeImage')

    # find PyOpenGL directory (cannot be in a zip file)
    opengl_location = OpenGL.__file__
    substring = os.path.join('OpenGL','')
    if substring not in opengl_location:
        raise Error('unable to extract PyOpenGL3 location from OpenGL module\'s __file__ member: '+repr(opengl_location))
    index = opengl_location.index(substring)-len(os.path.sep)
    opengl_location = opengl_location[0:index]

    print "Found PyOpenGL3 at:", opengl_location

    # allow msvcp90.dll to be included (from recipe in py2exe wiki's OverridingCriteriaForIncludingDlls)
    origIsSystemDLL = py2exe.build_exe.isSystemDLL
    def isSystemDLL(pathname):
        if os.path.basename(pathname).lower() in ("msvcp90.dll","umath.pyd"):
            return 0
        return origIsSystemDLL(pathname)
    py2exe.build_exe.isSystemDLL = isSystemDLL

    # run py2exe
    sys.argv.append('py2exe')
    setup(options = { 'py2exe' : {
                                  'optimize':2,
                                  'bundle_files':3, # no actual bundling
                                  'includes':['sip',
                                              ],
                                  'excludes':['Tkconstants',
                                              'Tkinter',
                                              'tcl',
                                              'OpenGL',
                                              ],
                                  }
                    },
          windows=[{'script':'LFDisplay.py'}],
          data_files = [('',['play.png','record.png','splash.png','pause.png',
                             'LFDisplay.vert',
                             'LFDisplay0.frag',
                             'LFDisplay1.frag',
                             'LFDisplay2.frag',
                             FreeImage_location]+msvcr_dlls+manifests)],
          )

    # copy OpenGL egg contents over
    src = os.path.join(opengl_location,'OpenGL')
    dst = os.path.join("dist",'OpenGL')
    print "Copying OpenGL egg from %s to %s" % (src, dst)
    shutil.copytree(src, dst)

    # create a zip file of the distribution
    print "Creating "+dist_name+".zip"
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk('dist'):
        zippath = dirpath.replace("dist", dist_name, 1)
        for filename in filenames:
            file_list.append((os.path.join(dirpath, filename),
                              os.path.join(zippath, filename)))
    z = zipfile.ZipFile(dist_name+'.zip', 'w', zipfile.ZIP_DEFLATED)
    for (src, dst) in file_list:
        z.write(src, dst)
    z.close()    
    
elif platform.system() == 'Darwin':
    import os
    import time
    import platform
    import sys
    import OpenGL
    import ctypes.util
    import shutil
    from setuptools import setup

    sys.argv.append('py2app')

    FreeImage_location = ctypes.util.find_library('libfreeimage.3.dylib')

    APP = ['LFDisplay.py']
    DATA_FILES = [('',['play.png','record.png','splash.png','pause.png'])]

    # figure out where to copy the PyOpenGL3 egg
    opengl_egg_location = OpenGL.__file__
    if '.egg/OpenGL/' not in opengl_egg_location:
        raise Error('unable to extract PyOpenGL3 egg location from OpenGL module\'s __file__ member: '+repr(opengl_egg_location))
    index = opengl_egg_location.index('.egg/OpenGL/') + 4
    opengl_egg_location = opengl_egg_location[0:index]
    opengl_egg_name = os.path.basename(opengl_egg_location)
    python_lib_folder = 'lib/python%d.%d' % (sys.version_info[0:2])
    # plist rewriting to enable the egg
    PLIST = {'PyResourcePackages':
             [python_lib_folder,
              os.path.join(python_lib_folder, 'lib-dynload'),
              os.path.join(python_lib_folder, 'site-packages.zip'),
              os.path.join(python_lib_folder, opengl_egg_name),
              ],
             }

    OPTIONS = {'argv_emulation': True,
               'plist' : PLIST,
               'includes' : ['sip', 'qcam'],
               'resources' : [FreeImage_location],
               'excludes' : ['OpenGL'] # disable copying over; will copy later
               }

    setup(
        app=APP,
        data_files=DATA_FILES,
        options={'py2app': OPTIONS},
        setup_requires=['py2app'],
        )

    # copy entire OpenGL library over
    src = opengl_egg_location
    dst = os.path.join("dist","LFDisplay.app","Contents","Resources",python_lib_folder, opengl_egg_name)
    print "Copying OpenGL egg from %s to %s" % (src, dst)
    shutil.copytree(src, dst)
    dmg_name = 'LFDisplay-OSX-'+platform.machine()+'-'+time.strftime('%Y%m%d',time.localtime(time.time()))+'.dmg'
    vol_name = 'LFDisplay for Mac OSX'
    exec_str = 'hdiutil create -ov -fs HFS+ -srcfolder dist -volname "%s" "%s"' % (vol_name,dmg_name)
    print exec_str
    os.system(exec_str)
else:
    print >> sys.stderr, 'Packaging not supported on '+platform.system()
