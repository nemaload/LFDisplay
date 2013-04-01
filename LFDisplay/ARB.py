"""
OpenGL ARB extensions we will be using

ext_names is a list of extensions we want to use

init() in this module needs to be called once we
have a valid OpenGL context
It contains code that reloads PyOpenGL3 so that
it can get valid function pointers in Windows
(this was a hack that was needed in early
versions of PyOpenGL3; not sure if it's still
needed but right now it doesn't break things)

Afterwards, extension functions can be called
from this module, e.g.

import ARB
ARB.init()
program = ARB.glCreateProgramObjectARB()
"""

import sys

class Error(Exception):
    pass

# for GLSL
ext_names = [
    'GL_ARB_fragment_shader',
    'GL_ARB_vertex_shader',
    'GL_ARB_shader_objects',
    'GL_ARB_multitexture',
    'GL_ARB_texture_non_power_of_two'
]

def ext_to_module(extension_name):
    """Convert an extension name into a module name"""
    parts = extension_name.split('_')
    module_name = ''.join(['OpenGL.',parts[0],'.',parts[1],'.','_'.join(parts[2:])])
    return module_name

def ext_to_init_name(extension_name):
    parts = extension_name.split('_')
    return ''.join([parts[0].lower(), 'Init'] +
                   [x.capitalize() for x in parts[2:]] +
                   [parts[1]])

module_names = [ext_to_module(x) for x in ext_names]

init_names = [ext_to_init_name(x) for x in ext_names]

def init():
    # import the modules into this module
    for module_name in module_names:
        # hack for PyOpenGL 3.0.0a5

        # reload the raw module
        module_components = module_name.split('.')
        raw_module_components = [module_components[0], 'raw'] + module_components[1:]
        raw_module_name = '.'.join(raw_module_components)
        if raw_module_name in sys.modules:
            # print 'Reloading ' + raw_module_name
            # print sys.modules[raw_module_name]
            sys.modules[raw_module_name] = reload(sys.modules[raw_module_name])
        # reload the actual module
        module = __import__(module_name,globals(),locals(),['*'])
        module = reload(module)
        # print '---'+module_name+'---'
        for member in [x for x in dir(module) if not x.startswith('_')]:
            globals()[member] = getattr(module,member)
            # print member + ' <- ' + repr(getattr(module,member))
        sys.modules[module_name] = module
    # initialize these extensions
    missing = []
    for ext_name,init_name in zip(ext_names,init_names):
        if not eval(init_name+'()'):
            missing.append(ext_name)
    if missing:
        raise Error('\n\t'.join(['Missing the following extensions:']+missing))
    # print '$$$'+str(bool(glCreateProgramObjectARB))
