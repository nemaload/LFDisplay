"""
An OpenGL based QWidget for displaying images from the camera
"""

from PyQt4 import QtCore, QtGui, QtOpenGL
# custom importer for ARB functions
import ARB
from OpenGL import GL

import math
import numpy
import Queue
import time
import sys
import os

import aperture
import gui

class Error(Exception):
    pass

# this needs to be disabled for proper focused image calculation 
ENABLE_PERSPECTIVE=False

APWIDTH = 16

_vertex = """

void main() {
  gl_TexCoord[0] = gl_MultiTexCoord0;
  gl_Position = ftransform();
  gl_FrontColor = gl_Color;
  gl_BackColor = gl_Color;
}

"""

# different fragment shader routines

_fragments = [
    # image display
    """
    uniform sampler2D tex;
    uniform vec4 gain;
    uniform vec4 offset;
    uniform vec4 gamma;
    
    void main(void) {
      gl_FragColor = texture2D(tex, gl_TexCoord[0].st);
      gl_FragColor = (gain*gl_FragColor+offset+vec4(gl_Color.rgb,1.0))*gl_Color.a;
      gl_FragColor = pow(gl_FragColor, gamma);
    }
    """,
    # light field (for pinhole rendering)
    """
    uniform sampler2D tex;
    uniform vec4 gain;
    uniform vec4 offset;
    uniform vec4 gamma;
    uniform mat2 rectLinear;
    uniform vec2 rectOffset;
    uniform vec4 normalDim;
    uniform mat4 RTM;

    void main(void) {
      vec2 coord, coord0, coord1, coord2, coord3, coord4;
      vec4 fracts;
      vec4 xyuv;
      vec4 color;
      xyuv = RTM*gl_TexCoord[0];
      if(all(bvec4(lessThanEqual(abs(xyuv.pq),vec2(0.5,0.5)),lessThanEqual(abs(xyuv.st),vec2(0.5,0.5))))) {
        coord = xyuv.st * normalDim.st;
        coord0 = floor(coord) + xyuv.pq;
        coord1 = rectLinear*coord0+rectOffset;
        coord2 = rectLinear*(coord0 + vec2(0.0,1.0))+rectOffset;
        coord3 = rectLinear*(coord0 + vec2(1.0,1.0))+rectOffset;
        coord4 = rectLinear*(coord0 + vec2(1.0,0.0))+rectOffset;
        fracts = vec4(coord-floor(coord),floor(coord)+vec2(1.0,1.0)-coord);
        color = fracts.z*(fracts.w*texture2D(tex, coord1)+
                          fracts.y*texture2D(tex, coord2)) +
          fracts.x*(fracts.y*texture2D(tex, coord3)+
                    fracts.w*texture2D(tex, coord4));
      }
      gl_FragColor = pow(vec4(gain.rgb*color.rgb+gl_Color.rgb+offset.rgb,gl_Color.a*gain.a*color.a),gamma);
    }
    """,
    # light field (for finite aperture rendering)
    """
    uniform sampler2D tex;
    uniform vec4 gain;
    uniform vec4 offset;
    uniform vec4 gamma;
    uniform mat2 rectLinear;
    uniform vec2 rectOffset;
    uniform vec4 normalDim;
    uniform mat4 RTM;
    uniform sampler2D aptex; // aperture texture
    uniform vec2 apertureScale; // how much scaling to multiply for aperture

    void main(void) {
      vec2 coord, coord0, coord1, coord2, coord3, coord4;
      vec2 uv;
      vec4 fracts;
      vec4 xyuv;
      vec4 aperture;
      int x, y;
      vec4 color;

      color = vec4(0.0,0.0,0.0,1.0);

      for(y = 0 ; y < %d ; y += 1) {
        for(x = 0 ; x < %d ; x += 1) {
          aperture = texture2D(aptex,vec2((float(x)+0.5)/%d.0,(float(y)+0.5)/%d.0));
          // comment: multiplying by 4th channel in aperture to obtain proper scaling and ignoring aperture component in input coordinate for abbe sine condition reasons
          xyuv = RTM*vec4(gl_TexCoord[0].xy, apertureScale*aperture.w*(aperture.xy-vec2(0.5,0.5)));
          if(aperture.z == 0.0) break;
          if(all(bvec4(lessThan(abs(xyuv.pq),vec2(0.5,0.5)),lessThan(abs(xyuv.st),vec2(0.5,0.5))))) {
            coord = xyuv.st * normalDim.st;
            // assume that we have no perspective effect, so use the
            // aperture value from the aperture texture instead of the
            // returned result (xyuv.pq)
            coord0 = floor(coord) + aperture.xy-vec2(0.5,0.5);
            coord1 = rectLinear*coord0+rectOffset;
            coord2 = rectLinear*(coord0 + vec2(0.0,1.0))+rectOffset;
            coord3 = rectLinear*(coord0 + vec2(1.0,1.0))+rectOffset;
            coord4 = rectLinear*(coord0 + vec2(1.0,0.0))+rectOffset;
            fracts = vec4(coord-floor(coord),floor(coord)+vec2(1.0,1.0)-coord);
            color += aperture.z*
              (fracts.z*(fracts.w*texture2D(tex, coord1)+
                         fracts.y*texture2D(tex, coord2)) +
               fracts.x*(fracts.y*texture2D(tex, coord3)+
                         fracts.w*texture2D(tex, coord4)));
          }
        }
        if(aperture.z == 0.0) break;
      }
      gl_FragColor = pow(vec4(  1.0  *gain.rgb*color.rgb+gl_Color.rgb+offset.rgb,gl_Color.a*gain.a*color.a),gamma);
    }
    """
]

# enable shader override

try:
    f=open("LFDisplay.vert")
    _vertex = f.read()
    f.close()
except Exception:
    pass

# enable fragment override

for i in range(3):
    try:
        f=open("LFDisplay%d.frag" % i)
        _fragments[i] = f.read()
        f.close()
    except Exception:
        pass

# do the APWIDTH replacement

_fragments[2] = _fragments[2].replace("%d",str(APWIDTH))

class ImagingDisplay(QtOpenGL.QGLWidget):
    def __init__(self, settings, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        # the display settings window
        self.displaySettings = None
        
        # set application settings
        self.settings = settings

        self.texture = 0
        # our current shader programs
        self.shaderPrograms = []

        # create a mutex for the state
        self.lock = QtCore.QMutex()

        # rate at which we want to run the timer
        self.timerInterval = self.settings.getInteger('display/timer_interval',20)

        # height and width of viewport
        self.width = 512
        self.height = 512

        # set dirty state
        self.dirty = False

        # start up an update timer
        self.timerId = self.startTimer(self.timerInterval)
        
        # whether GL is initialized
        self.initialized = False

        # start of drags
        self.dragStart = None
        self.dragStartUV = None

        # how big our source texture is
        self.textureSize = eval(self.settings.getString('display/texture_size','(512,512)'))

        # our new/current texture
        self.currentTexture = None # string of texture data
        self.textureType = None
        # how big our new texture is
        self.newTextureSize = self.textureSize

        # u and v panning parameters
        self.currentU = 0.0
        self.currentV = 0.0

        # ray-transfer matrix (row major)
        self.RTM = numpy.identity(4, dtype='float32')

        # image matrix (for the quad), (row major)
        self.imageMatrix = numpy.identity(4, dtype='float32')
        self.imageMatrix = numpy.array([[1., 0., 0., 0.,],
                                        [0., 1., 0., 0.,],
                                        [0., 0., 1., 0.,],
                                        [0., 0., 0., 1.,]],dtype='float32')
        self.imageMatrixInverse = numpy.identity(4, dtype='float32')
        self.imageMatrixInverse = numpy.array([[1., 0., 0., 0.,],
                                               [0., 1., 0., 0.,],
                                               [0., 0., 1., 0.,],
                                               [0., 0., 0., 1.,]],dtype='float32')
        # aperture samples (a list of tuples of (du,dv,alpha,scale))
        # scale is a scaling onto du and dv such that we have real slope
        # it's usually pretty small if z is in microns, since the slope
        # would be fraction of image per micron
        self.aperture = []
        for du in range(-4,5,1):
            for dv in range(-4,5,1):
                self.aperture.append((du*0.1,dv*0.1,1./81,1.0))
        # if aperture has changed (also if user pans image)
        self.apertureDirty = True 
        self.apertureList = None

        # queue of incoming frames
        self.inQueue = None

        # queue of outgoing frames
        self.outQueue = Queue.Queue()

        # initially frames not ready
        self.frameReady = False

        # current shader program being used
        self.shaderCurrent = self.settings.getInteger('display/shader_current',0)
        # the next shader program we want to use
        self.shaderNext = self.shaderCurrent

        # zoom level
        # This widget will emit a zoomChanged(float) signal
        # when the zoom level is changed
        self.zoomPower = self.settings.getFloat('display/zoom_power',0.0) # which power of two

        # actual width and height in pixels of the quad
        self.quadWidth = pow(2.0,self.zoomPower)*self.textureSize[0]
        self.quadHeight = pow(2.0,self.zoomPower)*self.textureSize[1]

        # offsets in pixels from center
        self.centerX = 0
        self.centerY = 0

        # the background color
        self.backgroundColor = eval(self.settings.getString('display/background_color','(0.5,0.5,0.5,1.0)'))

        # gamma level (used in shader)
        self.gamma = self.settings.getFloat('display/gamma',1.0)

        # gain level (used in shader)
        self.gain = self.settings.getFloat('display/gain',1.0)

        # minimum gain (not to be changed for now)
        self.gainMinimum = self.settings.getFloat('display/gain_minimum',1.0/255.0)

        # maximum gain (not to be changed for now)
        self.gainMaximum = self.settings.getFloat('display/gain_maximum',255.0)

        # frame gain level (set by loading a frame)
        self.frameGain = 16.0

        # current intensity level of the frame
        self.frameIntensity = 1.0

        # current desired intensity level of the frame
        self.desiredIntensity = 1.0

        # whether instant intensity feedback is enabled
        self.intensityAdjust = self.settings.getBool('display/intensity_adjust',False)

        # whether automatic gain control is enabled
        self.automaticGain = self.settings.getBool('display/automatic_gain', False)

        # automatic gain target level
        self.automaticGainTarget = self.settings.getFloat('display/automatic_gain_target', 0.9)

        # automatic gain fallback level (if saturates)
        self.automaticGainFallback = self.settings.getFloat('display/automatic_gain_fallback', 0.5)

        # offset level (used in shader)
        self.offset = self.settings.getFloat('display/offset',0.0)

        # Lenslet settings
        # these are all coordinates in the *displayed* image
        # using standard photoshop style image coordinates
        
        # offset from top left of image of the center lenslet center
        self.lensletOffset = eval(self.settings.getString('display/lenslet_offset','(255.5,255.5)'))
        # pixel delta for one lenslet over
        self.lensletHoriz = eval(self.settings.getString('display/lenslet_horiz','(17.0,0.0)'))
        # pixel delta for one lenslet down
        self.lensletVert = eval(self.settings.getString('display/lenslet_vert','(0.0,17.0)'))
        
        # whether to draw the lenslet grid
        self.drawGrid = self.settings.getBool('display/draw_grid',False)
        # draw center of lenslets
        # if 'lenslet', draw division of lenslets
        self.gridType = self.settings.getString('display/grid_type','center')
        # primary grid color
        self.gridColor = eval(self.settings.getString('display/grid_color','(0.0,1.0,0.0,0.5)'))
        # alternate grid color
        self.gridColorOther = eval(self.settings.getString('display/grid_color_other','(1.0,0.0,0.0,0.5)'))
        
        self.gridList = None # grid display list
        self.gridListDirty = True # whether we need to redraw grid

        # grab the saved optics recipe
        self.optics = {}
        self.optics['pitch'] = self.settings.getFloat('optics/pitch',125.0)
        self.optics['flen'] = self.settings.getFloat('optics/flen',2500.0)
        self.optics['mag'] = self.settings.getFloat('optics/mag',40.0)
        self.optics['abbe'] = self.settings.getBool('optics/abbe',True)
        self.optics['na'] = self.settings.getFloat('optics/na',0.95)
        self.optics['medium'] = self.settings.getFloat('optics/medium',1.0)

    def timerEvent(self, event):
        '''
        Call the OpenGL update function if necessary
        '''
        self.lock.lock()
        dirty = self.dirty or self.frameReady
        self.dirty = False
        self.lock.unlock()
        if dirty:
            self.updateGL()

    def setTimerInterval(self, rate):
        '''
        A slot to set a new timer interval corresponding to rate Hz
        '''
        newInterval = int(1000.0/rate+0.5)
        self.lock.lock()
        self.timerInterval = newInterval
        self.killTimer(self.timerId)
        self.timerId = self.startTimer()
        self.lock.unlock()

    def setQuadSize(self, quadWidth, quadHeight, center=None):
        """A slot to set the width and height of the quadrilateral we want"""
        self.lock.lock()
        if not center:
            center = (-self.centerX+self.width/2, self.height/2+self.centerY)
            
        # dx and dy in object space from center of quadrilateral, in pixels
        center = (center[0]-self.width/2,self.height/2-center[1])
        
        self.centerX += center[0]
        self.centerY += center[1]
        self.centerX /= self.quadWidth
        self.centerY /= self.quadHeight
        self.quadWidth = quadWidth
        self.quadHeight = quadHeight
        self.centerX *= self.quadWidth
        self.centerY *= self.quadHeight
        self.centerX -= center[0]
        self.centerY -= center[1]
        self.lock.unlock()
        self.resizeGL()

    def setQuadZoom(self, zoomLevel, center=None):
        """A slot to set the actual zoom level"""
        normalWidth, normalHeight = self.normalSize()
        self.setQuadSize(normalWidth * zoomLevel,
                         normalHeight * zoomLevel,
                         center)
        
    def setUV(self, u=0.0, v=0.0):
        """A slot to set the U,V coordinates (viewing direction)"""
        self.lock.lock()
        self.currentU = u
        self.currentV = v
        absSlope = (self.currentU*self.currentU + self.currentV*self.currentV)**0.5
        clampedAbsSlope = min(0.5, absSlope, self.maxNormalizedSlope())
        if absSlope and clampedAbsSlope < absSlope: 
            self.currentU *= clampedAbsSlope/absSlope
            self.currentV *= clampedAbsSlope/absSlope
        self.dirty = True
        self.apertureDirty = True
        self.lock.unlock()

    def setUVDelta(self, du, dv):
        "A slot to change the U,V coordinates (viewing direction)"
        self.lock.lock()
        self.currentU += du
        self.currentV += dv
        absSlope = (self.currentU*self.currentU + self.currentV*self.currentV)**0.5
        clampedAbsSlope = min(0.5, absSlope, self.maxNormalizedSlope())
        if absSlope and clampedAbsSlope < absSlope: 
            self.currentU *= clampedAbsSlope/absSlope
            self.currentV *= clampedAbsSlope/absSlope
        self.dirty = True
        self.apertureDirty = True
        self.lock.unlock()
        
    def initializeGL(self):
        """Initialize the GL environment we want"""
        if self.texture: return
        # grab a texture id
        self.texture, self.apertureTexture = [int(x) for x in GL.glGenTextures(2)]

        # set GL options
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glEnable(GL.GL_DEPTH_TEST)
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glFlush()

        # initialize our extensions
        try:
            ARB.init()
        except Exception, e:
            QtGui.QMessageBox.critical(self, 'Unable to initialize video hardware', 'Unable to initialize video hardware correctly.  Program behavior may be erratic.  Reason:\n'+str(e))
        
        # shaders
        for _fragment in _fragments:
            try:
                program = self.loadShadersText(_vertex,_fragment)
                self.shaderPrograms.append(program)
            except Exception, e:
                print >> sys.stderr, e
                if 'description' in dir(e):
                    print >> sys.stderr, e.description
        self.setShader(0)

        # create a display list to draw a circle with radius 0.5
        self.circleList = GL.glGenLists(1)
        segments = 64
        GL.glNewList(self.circleList, GL.GL_COMPILE)
        GL.glBegin(GL.GL_LINE_LOOP)
        for i in range(segments):
            GL.glVertex2f(0.5*math.cos(2.0*math.pi*i/segments),
                          0.5*math.sin(2.0*math.pi*i/segments))
        GL.glEnd()
        GL.glEndList()

    def resizeGL(self, width=None, height=None):
        '''
        Called when widget is resized
        '''
        self.lock.lock()
        if not width or not height:
            width = self.width
            height = self.height
        zoom = pow(2.0,self.zoomPower)
        self.centerX = max(min(self.centerX,width/2+self.quadWidth/2-64),-width/2-self.quadWidth/2+64)
        self.centerY = max(min(self.centerY,height/2+self.quadHeight/2-64),-height/2-self.quadHeight/2+64)
        left = self.centerX
        top = self.centerY
        self.lock.unlock()
        # set the basic view
        GL.glViewport(0,0,width,height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(math.floor(-0.5*width+left+0.5)/self.quadWidth,
                   math.floor(0.5*width+left+0.5)/self.quadWidth,
                   math.floor(-0.5*height+top+0.5)/self.quadHeight,
                   math.floor(+0.5*height+top+0.5)/self.quadHeight,
                   -1.0,1.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glFlush()
        # recalculate the texture coordinates
        self.lock.lock()
        self.width = width
        self.height = height
        self.dirty = True
        self.lock.unlock()
        # tell listeners about initial zoom
        self.emit(QtCore.SIGNAL('zoomChanged(float)'),zoom)

    def getGridMatrix(self, gridType='center'):
        "Calculate the grid matrix"
        # convert to pixel units in world space
        normLensletHorizX = 1.0*self.lensletHoriz[0]
        normLensletHorizY = -1.0*self.lensletHoriz[1]
        
        normLensletVertX = 1.0*self.lensletVert[0]
        normLensletVertY = -1.0*self.lensletVert[1]
        
        # convert offsets to pixel units in world space
        offsetX = self.lensletOffset[0] - (self.textureSize[0]-1)*0.5
        offsetY = (self.textureSize[1]-1)*0.5 - self.lensletOffset[1]
        if gridType == 'center':
            pass
        elif gridType == 'lenslet':
            offsetX += normLensletHorizX*0.5 + normLensletVertX*0.5
            offsetY += normLensletHorizY*0.5 + normLensletVertY*0.5
            
        # grid matrix is in world units (1.0 is width of texture, etc.)
        gridMatrix = [ normLensletHorizX/self.textureSize[0],
                       normLensletHorizY/self.textureSize[1],
                       0.0,
                       0.0,
                       normLensletVertX/self.textureSize[0],
                       normLensletVertY/self.textureSize[1],
                       0.0,
                       0.0,
                       0., # no transfer to z
                       0.,
                       0.,
                       0.,
                       offsetX/self.textureSize[0],
                       offsetY/self.textureSize[1],
                       0.0,
                       1.0,
                    ]

        return gridMatrix
        
    def paintGL(self):
        """Paint the screen"""
        start = time.time()
        # bind new texture if necessary
        self.lock.lock()
        # whether to load textures
        frameReady = self.frameReady
        self.frameReady = False
        # whether to change shader
        changeShader = self.shaderCurrent != self.shaderNext
        shaderNext = self.shaderNext
        self.lock.unlock()

        if changeShader:
            self.setShader(shaderNext)
            
        if frameReady:
            self.loadCurrentTexture()

        if shaderNext == 0:
            self.paintGLImage()
        elif shaderNext == 1:
            self.paintGLLFPinhole()
        elif shaderNext == 2:
            self.paintGLLFAperture()
        end = time.time()
        #if end-start:
        #    print '%.3f fps' % (1.0/(end-start))
        #else:
        #    print '--- fps'

    def calculateAperture(self, aperture):
        """
        Calculate the aperture uv scaling according to the Abbe sine condition
        and add in the uv offset as well
        """
        result = []
        apertureScale = 0.0
        M = self.recipe()['mag']
        inverseSpatialPixel = 1.0*M / self.recipe()['pitch']
        M21 = M*M - 1.0
        abbe = self.recipe()['abbe']
        # scale for microlens array
        baseScale = 1.0*self.recipe()['pitch'] / self.recipe()['flen']
        skip=0
        for (u,v,weight) in aperture:
            u += self.currentU
            v += self.currentV
            # starting absolute slope
            r = (u*u+v*v)**0.5
            if r == 0:
                result.append((0.5,0.5,weight,0))
                continue
            absSlopeMicrolens = r * baseScale
            # apply magnification
            s = absSlopeMicrolens
            s2 = absSlopeMicrolens*absSlopeMicrolens
            absSlopeMagnified = M * s
            if abbe: # abbe sine condition correction
                if 1-M21*s2 > 0:
                    absSlopeMagnified /= (1-M21*s2)**0.5
                else:
                    # angle too large, so skip
                    continue
            # convert to spatial pixels per micron z
            absSlopeMagnified2 = absSlopeMagnified * inverseSpatialPixel
            # figure out scaling
            scale = absSlopeMagnified2 / r
            # record maximum scale for later scaling
            apertureScale = max(apertureScale, scale)
            result.append((u+0.5,v+0.5,weight,scale))
        # apply scale and serialize the results
        result2 = []
        if apertureScale:
            for (u,v,weight,scale) in result:
                result2.extend((u,v,weight,scale/apertureScale))
        else:
            result2 = result
        return apertureScale,result2
       
    def paintGLLFAperture(self):
        "Paint a quad for the light field"
        
        GL.glDepthFunc(GL.GL_ALWAYS)
        GL.glClearColor(*self.backgroundColor)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # set a transform for the quad
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()

        # set the aperture texture
        ARB.glActiveTextureARB(ARB.GL_TEXTURE1_ARB)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.apertureTexture)
        if self.apertureDirty:
            apertureScale, apertureValues = self.calculateAperture(self.aperture)
            # set the aperture scaling
            location = ARB.glGetUniformLocationARB(self.shaderPrograms[2],
                                                   'apertureScale')
            if location >= 0:
                ARB.glUniform2fARB(location, apertureScale/self.normalSize()[0], apertureScale/self.normalSize()[1])
                GL.glFlush()       

            # now that we have the values, convert to float32
            apertureArray = numpy.array(apertureValues,dtype='float32')
            # make into string for texture
            apertureImage = apertureArray.tostring()
            if len(apertureImage) < APWIDTH*APWIDTH*4*4:
                apertureImage += '\x00' * (APWIDTH*APWIDTH*4*4-len(apertureImage))
            apertureImage = apertureImage[:APWIDTH*APWIDTH*4*4]
            # load the texture
            internalFormats = []
            try:
                internalFormats.append(GL.GL_RGBA16F_ARB)
            except Exception:
                pass
            try:
                internalFormats.append(GL.GL_RGBA32F_ARB)
            except Exception:
                pass
            try:
                internalFormats.append(GL.GL_RGBA16)
            except Exception:
                pass
            try:
                internalFormats.append(GL.GL_RGBA)
            except Exception:
                pass
            # try different internal formats, starting with the best one
            for internalFormat in internalFormats:
                try:
                    GL.glTexImage2D(GL.GL_TEXTURE_2D,
                                    0, internalFormat,
                                    APWIDTH, APWIDTH,
                                    0, GL.GL_RGBA, GL.GL_FLOAT,
                                    apertureImage)
                    GL.glFlush()
                    break
                except Exception:
                    pass
            GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D,
                               GL.GL_TEXTURE_MAG_FILTER,
                               GL.GL_NEAREST)
            self.apertureDirty = False

        # bind to light field texture
        ARB.glActiveTextureARB(ARB.GL_TEXTURE0_ARB)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

        # set the ray transfer matrix
        location = ARB.glGetUniformLocationARB(self.shaderPrograms[2],
                                               'RTM')
        if location >= 0:
            ARB.glUniformMatrix4fvARB(location, 1, GL.GL_TRUE, self.RTM)

        GL.glFlush()

        # draw the quad for the light field
        GL.glBegin(GL.GL_QUADS)

        # top left
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord4f(-0.5,0.5,self.currentU,self.currentV)
        GL.glVertex3f(-0.5,0.5,0.0)

        # top right
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord4f(0.5,0.5,self.currentU,self.currentV)
        GL.glVertex3f(0.5,0.5,0.0)

        # bottom right
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord4f(0.5,-0.5,self.currentU,self.currentV)
        GL.glVertex3f(0.5,-0.5,0.0)

        # bottom left
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord4f(-0.5,-0.5,self.currentU,self.currentV)
        GL.glVertex3f(-0.5,-0.5,0.0)

        GL.glEnd()

        GL.glFlush()

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()

        # process stats of rendered image if necessary
        self.processStats()

    def paintGLLFPinhole(self):
        "Paint a quad for the light field"
        
        GL.glDepthFunc(GL.GL_ALWAYS)
        GL.glClearColor(*self.backgroundColor)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # set a transform for the quad
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()

        # bind to light field texture
        ARB.glActiveTextureARB(ARB.GL_TEXTURE0_ARB)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

        # set the ray transfer matrix
        location = ARB.glGetUniformLocationARB(self.shaderPrograms[1],
                                               'RTM')
        if location >= 0:
            ARB.glUniformMatrix4fvARB(location, 1, GL.GL_TRUE, self.RTM)

        GL.glFlush()

        # draw the quad for the light field
        GL.glBegin(GL.GL_QUADS)

        # top left
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord4f(-0.5,0.5,self.currentU,self.currentV)
        GL.glVertex3f(-0.5,0.5,0.0)

        # top right
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord4f(0.5,0.5,self.currentU,self.currentV)
        GL.glVertex3f(0.5,0.5,0.0)

        # bottom right
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord4f(0.5,-0.5,self.currentU,self.currentV)
        GL.glVertex3f(0.5,-0.5,0.0)

        # bottom left
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord4f(-0.5,-0.5,self.currentU,self.currentV)
        GL.glVertex3f(-0.5,-0.5,0.0)

        GL.glEnd()

        GL.glFlush()

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()

        # process stats of rendered image if necessary
        self.processStats()

    def paintGLImage(self):
        "Paint a quad for the image"
        
        GL.glDepthFunc(GL.GL_ALWAYS)
        GL.glClearColor(*self.backgroundColor)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # set a transform for the quad
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadMatrixf(self.imageMatrix.transpose())

        # set the texture
        ARB.glActiveTextureARB(ARB.GL_TEXTURE0_ARB)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

        # draw the quad
        GL.glBegin(GL.GL_QUADS)

        # top left
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord2f(0.0,0.0)
        GL.glVertex3f(-0.5,0.5,0.0)

        # top right
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord2f(1.0,0.0)
        GL.glVertex3f(0.5,0.5,0.0)

        # bottom right
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord2f(1.0,1.0)
        GL.glVertex3f(0.5,-0.5,0.0)

        # bottom left
        GL.glColor4f(0.0,0.0,0.0,1.0)
        GL.glTexCoord2f(0.0,1.0)
        GL.glVertex3f(-0.5,-0.5,0.0)

        GL.glEnd()
        GL.glFlush()
        GL.glFinish()
        
        # restore transformation
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()

        # process stats of rendered image if necessary
        self.processStats()

        # draw the overlay
        if self.drawGrid:
            # enable blending to make grid centers look nice
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
            GL.glEnable(GL.GL_BLEND)
            # don't draw over parts of the scene that have no image
            GL.glDepthFunc(GL.GL_EQUAL)
            # save the modelview matrix
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glPushMatrix()

            # create a display list if necessary
            if self.gridListDirty:
                if not self.gridList:
                    # create a new display list if necessary
                    self.gridList = GL.glGenLists(1)
                # compile the list
                GL.glNewList(self.gridList, GL.GL_COMPILE)
                GL.glBegin(GL.GL_LINES)
                # draw the horizontal lines
                for y in range(-self.textureSize[1],self.textureSize[1]):
                    GL.glVertex2f(-self.textureSize[0],y)
                    GL.glVertex2f(self.textureSize[0],y)
                # draw the vertical lines
                for x in range(-self.textureSize[0],self.textureSize[0]):
                    GL.glVertex2f(x,-self.textureSize[1])
                    GL.glVertex2f(x,self.textureSize[1])
                GL.glEnd()
                # finish compiling the list
                GL.glEndList()
                self.gridListDirty = False

            # MAIN GRID
            # load the grid matrix
            gridMatrix = self.getGridMatrix(self.gridType)
            GL.glLoadMatrixf(gridMatrix)

            # call our display list
            if self.gridType == 'center':
                GL.glColor4fv((self.gridColor[0],
                               self.gridColor[1],
                               self.gridColor[2],
                               self.gridColor[3]*0.5))
            else:
                GL.glColor4fv((self.gridColorOther[0],
                               self.gridColorOther[1],
                               self.gridColorOther[2],
                               self.gridColorOther[3]*0.5))
            GL.glCallList(self.gridList)

            # MAIN AXIS
            if self.gridType == 'center':
                GL.glColor4fv(self.gridColor)
            else:
                GL.glColor4fv(self.gridColorOther)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex2f(-self.textureSize[0],0)
            GL.glVertex2f(self.textureSize[0],0)
            GL.glVertex2f(0,-self.textureSize[1])
            GL.glVertex2f(0,self.textureSize[1])
            if self.gridType == 'lenslet':
                GL.glVertex2f(-self.textureSize[0],-1)
                GL.glVertex2f(self.textureSize[0],-1)
                GL.glVertex2f(-1,-self.textureSize[1])
                GL.glVertex2f(-1,self.textureSize[1])
            GL.glEnd()

            # CENTRAL LENSLET CIRCLE
            maxu = self.maxNormalizedSlope()
            if maxu:
                # only draw if correct NA has been specified
                if self.gridType == 'center':
                    GL.glColor4fv((self.gridColorOther[0],
                                   self.gridColorOther[1],
                                   self.gridColorOther[2],
                                   self.gridColorOther[3]*0.5))
                else:
                    GL.glColor4fv(self.gridColorOther)
                GL.glLoadMatrixf(self.getGridMatrix('lenslet'))
                # the circle needs to appear at position (-0.5,-0.5) in this
                # transform matrix, with proper scaling
                GL.glTranslatef(-0.5,-0.5,0.0)
                GL.glScalef(2.0*maxu,2.0*maxu,1.0)
                GL.glCallList(self.circleList)

            # ALTERNATE GRID
            # draw the other grid pattern with more transparency
            gridMatrixOther = self.getGridMatrix(['center','lenslet'][self.gridType == 'center'])
            GL.glLoadMatrixf(gridMatrixOther)
            if self.gridType == 'center':
                GL.glColor4fv((self.gridColorOther[0],
                               self.gridColorOther[1],
                               self.gridColorOther[2],
                               self.gridColorOther[3]*0.5))
            else:
                GL.glColor4fv((self.gridColor[0],
                               self.gridColor[1],
                               self.gridColor[2],
                               self.gridColor[3]*0.5))

            # call our display list
            GL.glCallList(self.gridList)
            
            # restore the modelview matrix
            GL.glPopMatrix()
            # disable blending
            GL.glDisable(GL.GL_BLEND)
            # disable depth test
            GL.glDepthFunc(GL.GL_ALWAYS)

    def loadShadersText(self, vertexProgram, fragmentProgram):
        '''
        The actual implementation that loads a vertex and/or fragment shader
        '''
        program = ARB.glCreateProgramObjectARB()
        vertexShader = fragmentShader = None
        if vertexProgram:
            vertexShader = ARB.glCreateShaderObjectARB(ARB.GL_VERTEX_SHADER_ARB)
            ARB.glShaderSourceARB(vertexShader, [vertexProgram])
            ARB.glCompileShaderARB(vertexShader)
            if not ARB.glGetObjectParameterivARB(vertexShader, ARB.GL_OBJECT_COMPILE_STATUS_ARB):
                log = ARB.glGetInfoLogARB(vertexShader)
                raise Error('Error compiling vertex shader:\n'+log)
            ARB.glAttachObjectARB(program,vertexShader)
            GL.glFlush()
        if fragmentProgram:
            fragmentShader = ARB.glCreateShaderObjectARB(ARB.GL_FRAGMENT_SHADER_ARB)
            ARB.glShaderSourceARB(fragmentShader, [fragmentProgram])
            ARB.glCompileShaderARB(fragmentShader)
            if not ARB.glGetObjectParameterivARB(fragmentShader, ARB.GL_OBJECT_COMPILE_STATUS_ARB):
                log = ARB.glGetInfoLogARB(fragmentShader)
                raise Exception('Error compiling fragment shader:\n'+log)
            ARB.glAttachObjectARB(program,fragmentShader)
            GL.glFlush()
        ARB.glLinkProgramARB(program)
        #log = ARB.glGetInfoLogARB(program)
        #if log:
        #    raise Error('Error linking shaders:\n'+log)
        GL.glFlush()
        return program

    def emitDisplayModeChanged(self, num):
        self.emit(QtCore.SIGNAL('displayModeChanged(int)'), num)

    def processDisplayModeChanged(self, num):
        """
        Process a display mode change
        """
        if num == 0:
            self.setNextShader(0)
        elif num == 1 or num == 2:
            self.setNextShader(2)

    def setNextShader(self, num):
        """
        Set the shader to be used for the next frame
        """
        self.lock.lock()
        self.shaderNext = num
        self.dirty = True
        self.lock.unlock()

    def nextShader(self):
        """
        Get the shader to be used for the next frame
        """
        self.lock.lock()
        result = self.shaderNext
        self.lock.unlock()
        return result

    def setShader(self, num):
        """
        A slot to change the shader being used
        """
        if num >= len(self.shaderPrograms):
            return
        ARB.glUseProgramObjectARB(self.shaderPrograms[num])
        GL.glFlush()
        self.lock.lock()
        self.shaderCurrent = num
        zoomPower = self.zoomPower
        self.lock.unlock()
        self.setShaderPostprocess()
        self.setShaderTextures()
        if num in [1,2]:
            self.setRectification()
            self.setNormalDimensions()
        self.setQuadZoom(math.pow(2,zoomPower))
        self.settings.setValue('display/shader_current',num)
        self.emit(QtCore.SIGNAL('shaderChanged(int)'), num)

    def setShaderTextures(self):
        """
        Set the correct texture units for the shader
        """
        self.lock.lock()
        if not self.shaderPrograms or self.shaderCurrent >= len(self.shaderPrograms):
            self.lock.unlock()
            return
        shaderProgram = self.shaderPrograms[self.shaderCurrent]
        self.lock.unlock()
        location = ARB.glGetUniformLocationARB(shaderProgram,
                                               'tex')
        if location >= 0:
            ARB.glUniform1iARB(location, 0)
        location = ARB.glGetUniformLocationARB(shaderProgram,
                                               'aptex')
        if location >= 0:
            ARB.glUniform1iARB(location, 1)
        GL.glFlush()    

    def setRectification(self):
        """
        Calculate the rectification parameters from the
        values we have and send it
        """
        global numpy
        gridMatrix = numpy.array(self.getGridMatrix(),dtype='float32')
        gridMatrix.shape = (4,4)
        gridMatrix = gridMatrix.T
        # so that our grid basis vectors point down and right
        ySwapMatrix = numpy.array([[1., 0., 0., 0.,],
                                   [0., -1., 0., 0.,],
                                   [0., 0., 1., 0.,],
                                   [0., 0., 0., 1.,]],dtype='float32')
        # convert from world coordinate to texture coordinate
        worldToTextureMatrix = numpy.array([[1., 0., 0., 0.5],
                                            [0., -1., 0., 0.5],
                                            [0., 0., 1., 0.,],
                                            [0., 0., 0., 1.]],dtype='float32')
        temp=numpy.dot(self.imageMatrixInverse,numpy.dot(gridMatrix,ySwapMatrix))
        rectificationMatrix = numpy.dot(worldToTextureMatrix,temp)
        rectLinear = rectificationMatrix[0:2,0:2]
        rectOffset = rectificationMatrix[0:2,3]
        #print 'rectLinear: ',rectLinear
        #print 'rectOffset: ',rectOffset
        self.lock.lock()
        if not self.shaderPrograms or self.shaderCurrent >= len(self.shaderPrograms):
            self.lock.unlock()
            return
        shaderProgram = self.shaderPrograms[self.shaderCurrent]
        self.lock.unlock()
        location = ARB.glGetUniformLocationARB(shaderProgram,
                                               'rectLinear')
        if location >= 0:
            ARB.glUniformMatrix2fvARB(location, 1, GL.GL_FALSE, rectLinear.astype('float32').transpose())

        location = ARB.glGetUniformLocationARB(shaderProgram,
                                               'rectOffset')
        if location >= 0:
            ARB.glUniform2fARB(location, rectOffset[0], rectOffset[1])
        GL.glFlush()

    def setNormalDimensions(self):
        """
        Set the texture dimensions
        """
        self.lock.lock()
        if not self.shaderPrograms or self.shaderCurrent >= len(self.shaderPrograms):
            self.lock.unlock()
            return
        shaderProgram = self.shaderPrograms[self.shaderCurrent]
        self.lock.unlock()
        location = ARB.glGetUniformLocationARB(shaderProgram,
                                               'normalDim')
        normalSize = self.normalSize()

        if location >= 0:
            ARB.glUniform4fARB(location,
                               normalSize[0],
                               normalSize[1],
                               1./normalSize[0],
                               1./normalSize[1])
        GL.glFlush()       

    def setMatrix(self,
                  RTM=None,
                  imageMatrix=None):
        """
        Set one of the matrices
        """
        dirty = False
        if RTM is not None:
            self.RTM = RTM
            dirty = True
        if imageMatrix is not None:
            self.imageMatrix = imageMatrix
            dirty = True
        if dirty:
            self.dirty=True

    def recipe(self):
        """
        Return the optics recipe
        """
        return self.optics.copy()

    def setRecipe(self,
                  pitch=125.0,
                  flen=2500.0,
                  mag=40.0,
                  abbe=True,
                  na=0.95,
                  medium=1.0
                  ):
        """
        Set the optics recipe
        """
        self.optics['pitch'] = pitch
        self.optics['flen'] = flen
        self.optics['mag'] = mag
        self.optics['abbe'] = abbe
        self.optics['na'] = na
        self.optics['medium'] = medium
        self.settings.setValue('optics/pitch', pitch)
        self.settings.setValue('optics/flen', flen)
        self.settings.setValue('optics/mag', mag)
        self.settings.setValue('optics/abbe', abbe)
        self.settings.setValue('optics/na', na)
        self.settings.setValue('optics/medium', medium)

        # update min/max U/V
        maxNormalizedSlope = self.maxNormalizedSlope()
        self.minU = -maxNormalizedSlope
        self.maxU = maxNormalizedSlope
        self.minV = -maxNormalizedSlope
        self.maxV = maxNormalizedSlope
        
        self.dirty = True
        self.apertureDirty = True

    def maxNormalizedSlope(self):
        """
        Return the maximum slope afforded by the optical system
        0.5 means at the edge of a lenslet image
        """
        imagena = self.recipe()['na'] / self.recipe()['mag']
        if imagena < 1.0:
            ulenslope = 1.0 * self.recipe()['pitch'] / self.recipe()['flen']
            naslope = imagena / (1.0-imagena*imagena)**0.5
            return naslope / ulenslope
        else:
            return 0.0

    def emitPostprocessChanged(self):
        """
        Emit a signal when digital gain, offset, gamma is changed
        """
        self.emit(QtCore.SIGNAL('postprocessChanged()'))

    def postprocessSettings(self):
        """
        Return the current postprocess settings
        """
        result = {}
        self.lock.lock()
        try:
            result['gain'] = self.gain
            result['gain_minimum'] = self.gainMinimum
            result['gain_maximum'] = self.gainMaximum
            result['offset'] = self.offset
            result['gamma'] = self.gamma
            result['automatic_gain'] = self.automaticGain
            result['automatic_gain_target'] = self.automaticGainTarget
            result['automatic_gain_fallback'] = self.automaticGainFallback
            result['desired_intensity'] = self.desiredIntensity
            result['intensity_adjust'] = self.intensityAdjust
        finally:
            self.lock.unlock()
        return result

    def setShaderPostprocess(self):
        """
        Set shader settings for the postprocessing
        """
        self.lock.lock()
        try:
            if not self.shaderPrograms or self.shaderCurrent >= len(self.shaderPrograms):
                return
            shaderProgram = self.shaderPrograms[self.shaderCurrent]
            gain = self.gain
            gainAdjustment = self.frameGain
            if self.intensityAdjust:
                gainAdjustment *= (self.desiredIntensity/self.frameIntensity)
            offset = self.offset
            gamma = self.gamma
        finally:
            self.lock.unlock()
        gainLocation = ARB.glGetUniformLocationARB(shaderProgram, 'gain')
        offsetLocation = ARB.glGetUniformLocationARB(shaderProgram,
                                                    'offset')
        gammaLocation = ARB.glGetUniformLocationARB(shaderProgram,
                                                    'gamma')
        if gainLocation >= 0:
            ARB.glUniform4fARB(gainLocation,
                               gain*gainAdjustment,
                               gain*gainAdjustment,
                               gain*gainAdjustment, 1.0)
        if offsetLocation >= 0:
            ARB.glUniform4fARB(offsetLocation, offset, offset, offset, 0.0)
        if gammaLocation >= 0:
            ARB.glUniform4fARB(gammaLocation, gamma, gamma, gamma, 1.0)
        GL.glFlush()
        self.lock.lock()
        self.dirty = True
        self.lock.unlock()

    def setGamma(self, gamma):
        """
        Set the new gamma level
        """
        self.lock.lock()
        gammaChanged = (self.gamma != gamma)
        self.gamma = gamma
        self.lock.unlock()
        self.settings.setValue('display/gamma',gamma)
        if gammaChanged:
            self.emitPostprocessChanged()
            self.setShaderPostprocess()

    def setOffset(self, offset):
        """
        Set the new offset level
        """
        self.lock.lock()
        offsetChanged = (self.offset != offset)
        self.offset = offset
        self.lock.unlock()
        self.settings.setValue('display/offset',offset)
        if offsetChanged:
            self.emitPostprocessChanged()
            self.setShaderPostprocess()

    def setGain(self, gain):
        """
        Set the new gain level
        """
        self.lock.lock()
        gain = min(max(self.gainMinimum, gain),self.gainMaximum)
        gainChanged = (self.gain != gain)
        self.gain = gain
        self.lock.unlock()
        self.settings.setValue('display/gain',gain)
        if gainChanged:
            self.emitPostprocessChanged()
            self.setShaderPostprocess()

    def setDesiredIntensity(self, desiredIntensity):
        "A slot to set the desired display intensity of the frame"
        self.lock.lock()
        desiredIntensityChanged = (self.desiredIntensity != desiredIntensity)
        self.desiredIntensity = desiredIntensity
        self.lock.unlock()
        if desiredIntensityChanged:
            self.emitPostprocessChanged()
            self.setShaderPostprocess()

    def setIntensityAdjust(self, intensityAdjust):
        self.lock.lock()
        intensityAdjustChanged = (self.intensityAdjust != intensityAdjust)
        self.intensityAdjust = intensityAdjust
        self.lock.unlock()
        self.settings.setValue('display/intensity_adjust',intensityAdjust)
        if intensityAdjustChanged:
            self.emitPostprocessChanged()
            self.setShaderPostprocess()

    def setAutomaticGain(self, automaticGain):
        self.lock.lock()
        automaticGainChanged = (self.automaticGain != automaticGain)
        self.automaticGain = automaticGain
        self.lock.unlock()
        self.settings.setValue('display/automatic_gain',automaticGain)
        if automaticGainChanged:
            self.emitPostprocessChanged()
            self.setShaderPostprocess()

    def setGainOptions(self, automaticGainTarget, automaticGainFallback):
        self.lock.lock()
        oldGain = self.gain
        gainOptionsChanged = (automaticGainTarget != self.automaticGainTarget) or (automaticGainFallback != self.automaticGainFallback)
        self.automaticGainTarget = automaticGainTarget
        self.automaticGainFallback = automaticGainFallback
        self.lock.unlock()
        self.settings.setValue('display/automatic_gain_target',automaticGainTarget)
        self.settings.setValue('display/automatic_gain_fallback',automaticGainFallback)
        if gainOptionsChanged:
            self.setGain(oldGain)
            self.emitPostprocessChanged()
            self.setShaderPostprocess()

    def processStats(self):
        """
        Read the rendered image and do any processing on the stats
        """
        if not self.postprocessSettings()['automatic_gain']:
            return
        # figure out the rectangle that's the intersection of the
        # rendered quad with the viewport
        x1 = 0
        x2 = self.width
        y1 = 0
        y2 = self.height
        x1quad = -self.centerX + 0.5*self.width - 0.5*self.quadWidth
        x2quad = x1quad+self.quadWidth
        y1quad = -self.centerY + 0.5*self.height - 0.5*self.quadHeight
        y2quad = y1quad+self.quadHeight
        x1quad = max(x1,x1quad)
        x2quad = min(x2,x2quad)
        y1quad = max(y1,y1quad)
        y2quad = min(y2,y2quad)
        # trim off a border of 1 pixel thick to be safe
        x = round(x1quad)+1
        y = round(y1quad)+1
        width = round(x2quad-x1quad)-2
        height = round(y2quad-y1quad)-2
        # read the data
        GL.glReadBuffer(GL.GL_BACK)
        s = GL.glReadPixels(x, y, width, height, GL.GL_RGB,
                            GL.GL_UNSIGNED_BYTE)
        array = numpy.fromstring(s, 'B')
        t = GL.glReadPixels(x, y, 1, 1, GL.GL_ALPHA, GL.GL_UNSIGNED_BYTE)
        if t != '\x00':
            colorMax = array.max()
            # cap gain adjustment at 2x per cycle
            gainAdjustment = 240.0/max(colorMax,120.0)
            settings = self.postprocessSettings()

            currentGain = settings['gain']
            offset = settings['offset']
            gamma = settings['gamma']
            target = settings['automatic_gain_target']
            fallback = settings['automatic_gain_fallback']
            minGain = settings['gain_minimum']
            maxGain = settings['gain_maximum']
            
            if colorMax == 255:
                # fallback fast
                colorMax = 1.0
                colorDesired = fallback
            else:
                colorMax = colorMax / 255.0
                colorDesired = target
            

            rawColorMax = (colorMax ** (1.0/gamma) - offset) / currentGain
            temp = (colorDesired ** (1.0/gamma) - offset)

            if rawColorMax <= 0.0:
                return
            elif temp >= maxGain * rawColorMax:
                newGain = maxGain
            elif temp <= minGain * rawColorMax:
                newGain = minGain
            else:
                newGain = max(minGain, temp / max(rawColorMax, temp/maxGain))
            
            #print 'color max is: ', colorMax
            #print 'new gain: ', newGain

            self.setGain(newGain)
        
    def setQueue(self, queue):
        """
        Set the frame queue from which we are getting frames
        """
        self.inQueue = queue

    def setGrid(self,
                lensletOffset = None,
                lensletHoriz = None,
                lensletVert = None,
                drawGrid = None,
                gridType = None,
                gridColor = None,
                gridColorOther = None):
        """
        Set one or more of the grid parameters
        """
        dirty = False
        if None != lensletOffset:
            self.lensletOffset = lensletOffset
            self.settings.setValue('display/lenslet_offset',repr(lensletOffset))
            dirty = True
        if None != lensletHoriz:
            self.lensletHoriz = lensletHoriz
            self.settings.setValue('display/lenslet_horiz',repr(lensletHoriz))
            dirty = True
        if None != lensletVert:
            self.lensletVert = lensletVert
            self.settings.setValue('display/lenslet_vert',repr(lensletVert))
            dirty = True
        if None != drawGrid:
            self.drawGrid = drawGrid
            self.settings.setValue('display/draw_grid',drawGrid)
            dirty = True
        if None != gridType:
            self.gridType = gridType
            self.settings.setValue('display/grid_type',gridType)
            dirty = True
        if None != gridColor:
            self.gridColor = gridColor
            self.settings.setValue('display/grid_color',repr(gridColor))
            dirty = True
        if None != gridColorOther:
            self.gridColorOther = gridColorOther
            self.settings.setValue('display/grid_color_other',repr(gridColorOther))
            dirty = True
        if dirty:
            self.dirty = True
            if self.shaderCurrent in [1,2]:
                self.setRectification()
                self.setNormalDimensions()
            self.emit(QtCore.SIGNAL('lensletChanged()'))

    def newFrame(self):
        """
        A slot to indicate that a frame is ready on the input queue
        """
        if not self.inQueue:
            return
        try:
            frame = self.inQueue.get(False)
        except self.inQueue.Empty:
            # do nothing for now
            return
        # grab the texture
        newTextureSize = frame.width, frame.height
        currentTexture = frame.stringBuffer[:]
        textureFormat = str(frame.formatString)
        bits = frame.bits
        intensity = frame.intensity
        # done with frame
        self.outQueue.put(frame)
        self.emit(QtCore.SIGNAL('frameDone()'))
        # set the new texture
        self.lock.lock()
        self.newTextureSize = newTextureSize
        self.currentTexture = currentTexture
        self.textureFormat = textureFormat
        self.textureBits = bits # actual number of bits in frame
        self.newFrameIntensity = intensity # relative intensity of frame
        self.frameReady = True
        self.dirty = True
        self.lock.unlock()

    def loadCurrentTexture(self):
        """
        Load the current texture
        """
        frameGainChanged = False
        self.lock.lock()
        newTextureSize = self.newTextureSize
        currentTexture = self.currentTexture
        textureFormat = self.textureFormat
        bits = self.textureBits
        newFrameIntensity = self.newFrameIntensity
        self.lock.unlock()
        # collect the information we need from the frame
        if textureFormat == 'mono8':
            # use luminance8
            _internalformat = GL.GL_LUMINANCE8
            _format = GL.GL_LUMINANCE
            _type = GL.GL_UNSIGNED_BYTE
            frameGain = math.pow(2.0,8-bits)
            frameGainChanged = not(self.frameGain == frameGain)
            self.frameGain = frameGain
        elif textureFormat == 'mono16':
            # use luminance16
            _internalformat = GL.GL_LUMINANCE16
            _format = GL.GL_LUMINANCE
            _type = GL.GL_UNSIGNED_SHORT
            frameGain = math.pow(2.0,16-bits)
            frameGainChanged = not(self.frameGain == frameGain)
            self.frameGain = frameGain
        elif textureFormat == 'rgb24':
            # use rgb8
            _internalformat = GL.GL_RGB8
            _format = GL.GL_RGB
            _type = GL.GL_UNSIGNED_BYTE
            frameGain = math.pow(2.0,8-bits)
            frameGainChanged = not(self.frameGain == frameGain)
            self.frameGain = frameGain
        elif textureFormat == 'bgr24':
            # use rgb8
            _internalformat = GL.GL_RGB8
            _format = GL.GL_BGR
            _type = GL.GL_UNSIGNED_BYTE
            frameGain = math.pow(2.0,8-bits)
            frameGainChanged = not(self.frameGain == frameGain)
            self.frameGain = frameGain
        elif textureFormat == 'rgba32':
            # use rgb8
            _internalformat = GL.GL_RGBA8
            _format = GL.GL_RGBA
            _type = GL.GL_UNSIGNED_BYTE
            frameGain = math.pow(2.0,8-bits)
            frameGainChanged = not(self.frameGain == frameGain)
            self.frameGain = frameGain
        elif textureFormat == 'bgra32':
            # use rgb8
            _internalformat = GL.GL_RGBA8
            _format = GL.GL_BGRA
            _type = GL.GL_UNSIGNED_BYTE
            frameGain = math.pow(2.0,8-bits)
            frameGainChanged = not(self.frameGain == frameGain)
            self.frameGain = frameGain
        elif textureFormat == 'rgb48':
            # use rgb16
            _internalformat = GL.GL_RGB16
            _format = GL.GL_RGB
            _type = GL.GL_UNSIGNED_SHORT
            frameGain = math.pow(2.0,16-bits)
            frameGainChanged = not(self.frameGain == frameGain)
            self.frameGain = frameGain
        elif textureFormat == 'bgr48':
            # use rgb16
            _internalformat = GL.GL_RGB16
            _format = GL.GL_BGR
            _type = GL.GL_UNSIGNED_SHORT
            frameGain = math.pow(2.0,16-bits)
            frameGainChanged = not(self.frameGain == frameGain)
            self.frameGain = frameGain
        elif textureFormat == 'rgba64':
            # use rgb16
            _internalformat = GL.GL_RGBA16
            _format = GL.GL_RGBA
            _type = GL.GL_UNSIGNED_SHORT
            frameGain = math.pow(2.0,16-bits)
            frameGainChanged = not(self.frameGain == frameGain)
            self.frameGain = frameGain
        elif textureFormat == 'bgra64':
            # use rgb16
            _internalformat = GL.GL_RGBA16
            _format = GL.GL_BGRA
            _type = GL.GL_UNSIGNED_SHORT
            frameGain = math.pow(2.0,16-bits)
            frameGainChanged = not(self.frameGain == frameGain)
            self.frameGain = frameGain
        else:
            raise Error('Unsupported frame image format: '+textureFormat)
        _width, _height = newTextureSize
        _border = 0
        _level = 0
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glPixelStorei(GL.GL_UNPACK_SWAP_BYTES, 0)
        GL.glPixelStorei(GL.GL_UNPACK_ROW_LENGTH, 0)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D,
                        _level, _internalformat,
                        _width, _height,
                        _border, _format, _type,
                        currentTexture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_WRAP_S,
                           GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_WRAP_T,
                           GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameterfv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR,
                            [0.0, 0.0, 0.0, 1.0])

        self.lock.lock()
        changeZoom = self.textureSize != newTextureSize
        oldTextureSize = self.textureSize
        self.textureSize = newTextureSize
        frameIntensityChanged = self.frameIntensity != newFrameIntensity
        self.frameIntensity = newFrameIntensity
        self.lock.unlock()
        if changeZoom:
            # save settings
            self.settings.setValue('display/texture_size',repr(newTextureSize))
            # rescale the grid
            scale = (1.*newTextureSize[0]/oldTextureSize[0],
                     1.*newTextureSize[1]/oldTextureSize[1])
            lensletOffset = (self.lensletOffset[0]*scale[0],
                             self.lensletOffset[1]*scale[1])
            lensletHoriz = (self.lensletHoriz[0]*scale[0],
                             self.lensletHoriz[1]*scale[1])
            lensletVert = (self.lensletVert[0]*scale[0],
                             self.lensletVert[1]*scale[1])
            self.setGrid(lensletOffset=lensletOffset,
                         lensletHoriz=lensletHoriz,
                         lensletVert=lensletVert)
            self.gridListDirty = True # redo the grid
            self.setQuadZoom(math.pow(2,self.zoomPower))
            if self.shaderCurrent in [1,2]:
                self.setRectification()
                self.setNormalDimensions()
        if frameGainChanged or frameIntensityChanged:
            self.setShaderPostprocess()

    def normalSize(self):
        """
        Return the 1x zoom size of the quad, given the current shader state
        """
        self.lock.lock()
        textureSize = self.textureSize
        shaderCurrent = self.shaderCurrent
        horizX = self.lensletHoriz[0]
        vertY = self.lensletVert[1]
        offset = self.lensletOffset
        self.lock.unlock()
        if shaderCurrent in [0]:
            # image shaders
            return (1.0*math.floor(textureSize[0]), 1.0*math.floor(textureSize[1]))
        elif shaderCurrent in [1,2]:
            # prevent division by zero
            if horizX < 1 and horizX >= 0:
                horizX = 1.0
            elif horizX < 0 and horizX > -1:
                horizX = -1.0
            if vertY < 1 and vertY >= 0:
                vertY = 1.0
            elif vertY < 0 and vertY > -1:
                vertY = -1.0
            # calculate number of lenslets to the left and right
            leftLenslets = math.ceil(abs(offset[0] / horizX))
            rightLenslets = math.ceil(abs((textureSize[0]-offset[0])/horizX))
            # calculate the number of lenslets up and down
            upLenslets = math.ceil(abs(offset[1] / vertY))
            downLenslets = math.ceil(abs((textureSize[1]-offset[1]) / vertY))
            # return the extent that will cover both
            return (2*max(leftLenslets,rightLenslets),2*max(upLenslets,downLenslets))

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            # start a translation drag (on mac modifiers get set to something)
            self.dragStart = (event.x(),event.y())
        elif event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.NoModifier:
            # start a uv drag
            self.dragStartUV = (event.x(),event.y())
        else:
            QtOpenGL.QGLWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if self.dragStart:
            # continue a drag
            startX, startY = self.dragStart
            deltaX = event.x()-startX
            deltaY = event.y()-startY
            self.centerY += deltaY
            self.centerX -= deltaX
            self.dragStart = event.x(), event.y()
            self.resizeGL()
        if self.dragStartUV:
            # continue a UV drag
            startX, startY = self.dragStartUV
            deltaX = event.x()-startX
            deltaY = event.y()-startY
            w,h = self.normalSize()
            self.setUVDelta(-deltaX/w,deltaY/h)
            self.dragStartUV = event.x(), event.y()
            self.dirty = True
            self.apertureDirty = True
        QtOpenGL.QGLWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if self.dragStart:
            # end a drag
            startX, startY = self.dragStart
            deltaX = event.x()-startX
            deltaY = event.y()-startY
            self.centerY += deltaY
            self.centerX -= deltaX
            self.dragStart = None
            self.resizeGL()
        if self.dragStartUV:
            # end a UV drag
            startX, startY = self.dragStartUV
            deltaX = event.x()-startX
            deltaY = event.y()-startY
            w,h = self.normalSize()
            self.setUVDelta(-deltaX/w,deltaY/h)
            self.dragStartUV = None
            self.dirty = True
        QtOpenGL.QGLWidget.mouseReleaseEvent(self, event)

    def get(self, block=True, timeout=None):
        """
        Get a frame from the outgoing queue
        """
        return self.outQueue.get(block, timeout)

    def changeZoom(self, change, coords=None):
        "Change the amount of zoom"
        lastZoomPower = self.zoomPower
        self.zoomPower += change
        self.zoomPower = min(self.zoomPower, 5)
        self.zoomPower = max(self.zoomPower, -5)
        if self.zoomPower != lastZoomPower:
            # zoom changed
            zoom = math.pow(2.0,self.zoomPower)
            self.setQuadZoom(zoom, coords)
            self.emit(QtCore.SIGNAL('zoomChanged(float)'),zoom)
            self.settings.setValue('display/zoom_power',float(self.zoomPower))            
    def wheelEvent(self, event):
        "Zoom the image in and out"
        if event.modifiers() == QtCore.Qt.NoModifier:
            # process zoom
            if event.delta() > 0:
                self.changeZoom(1.0, (event.x(), event.y()))
            elif event.delta() < 0:
                self.changeZoom(-1.0, (event.x(), event.y()))
            else:
                QtOpenGL.QGLWidget.wheelEvent(self, event)
            
        else:
            QtOpenGL.QGLWidget.wheelEvent(self, event)


class DisplaySettings(QtGui.QWidget):
    """
    A window that has the various display-specific settings
    """
    def __init__(self, displayWindow, parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        self.displayWindow = displayWindow
        self.displayWindow.displaySettings = self

        self.gainOptionsGroup = QtGui.QGroupBox('Digital gain options', self)
        self.gainOptionsLayout = QtGui.QGridLayout(self.gainOptionsGroup)
        self.automaticGain = QtGui.QCheckBox('Automatic gain control')
        self.advancedOptions = QtGui.QCheckBox('Advanced options')
        self.automaticGainTarget = gui.SliderWidget(gui.LinearMap(0.0,1.0),
                                                    (float,lambda x:'%.3f'%x),
                                                    0.9,
                                                    'Automatic gain target intensity',
                                                    steps=999)
        self.automaticGainTarget.setToolTip('Automatic gain will attempt to make \nthe brightest pixel have this intensity')
        self.automaticGainFallback = gui.SliderWidget(gui.LinearMap(0.0,1.0),
                                                    (float,lambda x:'%.3f'%x),
                                                    0.5,
                                                    'Automatic gain saturation fallback intensity',
                                                    steps=999)
        self.automaticGainFallback.setToolTip('If a pixel hits intensity 1.0 (saturation), \nthe automatic gain will scale back the intensity to this')
        self.automaticGainTarget.setVisible(False)
        self.automaticGainFallback.setVisible(False)
        self.connect(self.automaticGain,
                     QtCore.SIGNAL('toggled(bool)'),
                     self.automaticGainToggled)
        self.connect(self.advancedOptions,
                     QtCore.SIGNAL('toggled(bool)'),
                     self.advancedOptionsToggled)
        self.connect(self.automaticGainTarget,
                     QtCore.SIGNAL('valueChanged()'),
                     self.gainOptionsChanged)
        self.connect(self.automaticGainFallback,
                     QtCore.SIGNAL('valueChanged()'),
                     self.gainOptionsChanged)
        self.gainOptionsLayout.addWidget(self.automaticGain, 0, 0)
        self.gainOptionsLayout.addWidget(self.advancedOptions, 0, 1)
        self.gainOptionsLayout.addWidget(self.automaticGainTarget, 1, 0, 1, 2)
        self.gainOptionsLayout.addWidget(self.automaticGainFallback, 2, 0, 1, 2)
        self.gainOptionsGroup.setLayout(self.gainOptionsLayout)

        settings = self.displayWindow.postprocessSettings()
        minGain = settings['gain_minimum']
        maxGain = settings['gain_maximum']
        self.gainGroup = gui.SliderWidget(gui.ExponentialMap(minGain,maxGain),
                                          (float,lambda x:'%.3g'%x),
                                          1.0,
                                          'Digital gain',
                                          steps=999,
                                          compact=True)
        self.connect(self.gainGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.gainChanged)

        self.offsetGroup = gui.SliderWidget(gui.LinearMap(-1.0,1.0),
                                            (float,lambda x:'%.3f'%x),
                                            0.0,
                                            'Digital offset',
                                            steps=999,
                                            compact=True)
        self.connect(self.offsetGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.offsetChanged)
        
        self.gammaGroup = gui.SliderWidget(gui.ExponentialMap(0.05,20.0),
                                           (float,lambda x:'%.3f'%x),
                                           1.0,
                                           'Digital gamma',
                                           steps=999,
                                           compact=True)

        self.connect(self.gammaGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.gammaChanged)


        self.buttonGroup = QtGui.QGroupBox('')
        self.resetPostprocess = QtGui.QPushButton('Reset gain, offset, gamma')
        self.buttonLayout = QtGui.QGridLayout(self.buttonGroup)
        self.buttonLayout.addWidget(self.resetPostprocess,0,0)
        self.buttonLayout.setColumnStretch(1,1)
        self.buttonGroup.setLayout(self.buttonLayout)

        self.connect(self.resetPostprocess,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.resetPostprocessSettings)
       
        self.gridTypeGroup = QtGui.QGroupBox('Grid type', self)
        self.gridNone = QtGui.QRadioButton('None')
        self.gridCenters = QtGui.QRadioButton('Lenslet centers')
        self.gridBoundaries = QtGui.QRadioButton('Lenslet boundaries')
        self.gridTypeLayout = QtGui.QGridLayout(self.gridTypeGroup)
        self.gridTypeLayout.setSpacing(0)
        self.gridTypeLayout.addWidget(self.gridNone,0,0)
        self.gridTypeLayout.addWidget(self.gridCenters,0,1)
        self.gridTypeLayout.addWidget(self.gridBoundaries,1,1)
        self.gridTypeGroup.setLayout(self.gridTypeLayout)

        self.gridColorGroup = QtGui.QGroupBox('Grid color', self)
        self.gridColorDisplay = QtGui.QLineEdit(' ')
        self.gridColorDisplay.setMaximumWidth(48)
        self.gridColorDisplay.setReadOnly(True)
        self.gridColorPalette = QtGui.QPalette()
        self.gridColorDisplay.setPalette(self.gridColorPalette)
        self.gridColorButton = QtGui.QPushButton('Centers...')
        self.gridColorOtherDisplay = QtGui.QLineEdit(' ')
        self.gridColorOtherDisplay.setMaximumWidth(48)
        self.gridColorOtherDisplay.setReadOnly(True)
        self.gridColorOtherPalette = QtGui.QPalette()
        self.gridColorOtherDisplay.setPalette(self.gridColorOtherPalette)
        self.gridColorOtherButton = QtGui.QPushButton('Borders...')
        self.gridOpacitySlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.gridOpacitySlider.setMinimum(0)
        self.gridOpacitySlider.setMaximum(100)
        self.gridOpacityDisplay = QtGui.QLabel('Opacity: 50%')
        self.gridOpacityOtherSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.gridOpacityOtherSlider.setMinimum(0)
        self.gridOpacityOtherSlider.setMaximum(100)
        self.gridOpacityOtherDisplay = QtGui.QLabel('Opacity: 50%')
        self.gridColorLayout = QtGui.QGridLayout(self.gridColorGroup)
        self.gridColorLayout.addWidget(self.gridColorDisplay,0,0)
        self.gridColorLayout.addWidget(self.gridColorButton,0,1)
        self.gridColorLayout.addWidget(self.gridColorOtherDisplay,0,2)
        self.gridColorLayout.addWidget(self.gridColorOtherButton,0,3)
        self.gridColorLayout.addWidget(self.gridOpacitySlider,1,0,1,2)
        self.gridColorLayout.addWidget(self.gridOpacityDisplay,2,0,1,2)
        self.gridColorLayout.addWidget(self.gridOpacityOtherSlider,1,2,1,2)
        self.gridColorLayout.addWidget(self.gridOpacityOtherDisplay,2,2,1,2)
        self.gridColorGroup.setLayout(self.gridColorLayout)

        self.settingsLayout = QtGui.QGridLayout(self)
        self.settingsLayout.addWidget(self.gainOptionsGroup,1,0)
        self.settingsLayout.addWidget(self.gainGroup,2,0)
        self.settingsLayout.addWidget(self.offsetGroup,3,0)
        self.settingsLayout.addWidget(self.gammaGroup,4,0)
        self.settingsLayout.addWidget(self.buttonGroup,5,0)
        self.settingsLayout.addWidget(self.gridTypeGroup,6,0)
        self.settingsLayout.addWidget(self.gridColorGroup,7,0)
        self.settingsLayout.setRowStretch(8,1)
        self.setLayout(self.settingsLayout)

        self.updateFromParent()

        self.connect(self.displayWindow,
                     QtCore.SIGNAL('postprocessChanged()'),
                     self.updateFromParent)

        self.connect(self.gridNone,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.gridTypeChanged)
        self.connect(self.gridCenters,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.gridTypeChanged)
        self.connect(self.gridBoundaries,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.gridTypeChanged)

        self.connect(self.gridColorButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.chooseGridColor)
        self.connect(self.gridColorOtherButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.chooseGridColorOther)
        self.connect(self.gridOpacitySlider,
                     QtCore.SIGNAL('valueChanged(int)'),
                     self.opacityChanged)
        self.connect(self.gridOpacityOtherSlider,
                     QtCore.SIGNAL('valueChanged(int)'),
                     self.opacityOtherChanged)

    def advancedOptionsToggled(self, b):
        self.automaticGainTarget.setVisible(b)
        self.automaticGainFallback.setVisible(b)

    def resetPostprocessSettings(self):
        self.displayWindow.setAutomaticGain(False)
        self.displayWindow.setGain(1.0)
        self.displayWindow.setOffset(0.0)
        self.displayWindow.setGamma(1.0)
        self.displayWindow.setGainOptions(0.9,0.5)
        self.updateFromParent()

    def updateFromParent(self):
        """
        Update settings display from actual display widget
        """
        shaderProgram = self.displayWindow.nextShader()
        # get postprocessSettings
        postprocess = self.displayWindow.postprocessSettings()
        # set the gain options
        self.automaticGain.setChecked(postprocess['automatic_gain'])
        self.automaticGainTarget.setValue(postprocess['automatic_gain_target'])
        self.automaticGainFallback.setValue(postprocess['automatic_gain_fallback'])
        # set the gain
        gain = postprocess['gain']
        self.gainGroup.setValue(gain)
        self.gainGroup.setEnabled(not postprocess['automatic_gain'])
        # set the offset
        offset = postprocess['offset']
        self.offsetGroup.setValue(offset)
        # set the gamma
        gamma = postprocess['gamma']
        self.gammaGroup.setValue(gamma)
        # update grid type from parent
        if not self.displayWindow.drawGrid:
            self.gridNone.setChecked(True)
        elif self.displayWindow.gridType == 'center':
            self.gridCenters.setChecked(True)
        elif self.displayWindow.gridType == 'lenslet':
            self.gridBoundaries.setChecked(True)
        # update grid color
        gridColorRGB = self.displayWindow.gridColor[0:3]
        self.gridColorPalette.setColor(QtGui.QPalette.Base,
                                       QtGui.QColor.fromRgbF(*gridColorRGB))
        self.gridColorDisplay.setPalette(self.gridColorPalette)
        gridColorOtherRGB = self.displayWindow.gridColorOther[0:3]
        self.gridColorOtherPalette.setColor(QtGui.QPalette.Base,
                                            QtGui.QColor.fromRgbF(*gridColorOtherRGB))
        self.gridColorOtherDisplay.setPalette(self.gridColorOtherPalette)
        opacity = int(self.displayWindow.gridColor[3] * 100 + 0.5)
        self.gridOpacitySlider.setValue(opacity)
        opacityOther = int(self.displayWindow.gridColorOther[3] * 100 + 0.5)
        self.gridOpacityOtherSlider.setValue(opacityOther)
        self.gridOpacityDisplay.setText('Opacity: %d%%' % self.gridOpacitySlider.value())
        self.gridOpacityOtherDisplay.setText('Opacity: %d%%' % self.gridOpacityOtherSlider.value())

    def automaticGainToggled(self, b):
        """
        The automatic gain control was toggled
        """
        self.displayWindow.setAutomaticGain(b)
        # disable gain slider if needed
        self.gainGroup.setEnabled(not b)

    def displayTypeChanged(self):
        """
        When the user selects a different display type
        """
        for i in range(len(self.displayTypes)):
            if self.displayTypes[i].isChecked():
                self.displayWindow.setNextShader(i)
                break
        self.updateFromParent()

    def gainChanged(self):
        """
        When the user selects a different gain
        """
        self.displayWindow.setGain(self.gainGroup.value())

    def offsetChanged(self):
        """
        When the user selects a different offset
        """
        self.displayWindow.setOffset(self.offsetGroup.value())

    def gammaChanged(self):
        """
        When the user adjusts the gamma
        """
        self.displayWindow.setGamma(self.gammaGroup.value())

    def gainOptionsChanged(self):
        """
        When the user adjusts advanced options for gain
        """
        self.displayWindow.setGainOptions(self.automaticGainTarget.value(),
                                          self.automaticGainFallback.value())

    def gridTypeChanged(self):
        if self.gridNone.isChecked():
            self.displayWindow.setGrid(drawGrid=False)
        elif self.gridCenters.isChecked():
            self.displayWindow.setGrid(drawGrid=True, gridType='center')
        elif self.gridBoundaries.isChecked():
            self.displayWindow.setGrid(drawGrid=True, gridType='lenslet')
        self.updateFromParent()

    def chooseGridColor(self):
        newColor = QtGui.QColorDialog.getColor()
        if newColor.isValid():
            # user selected a new color
            newColorRGB = newColor.getRgbF()
            curAlpha = self.gridOpacitySlider.value() / 100.0
            newColorRGBA = newColorRGB[0:3]+(curAlpha,)
            self.displayWindow.setGrid(gridColor=newColorRGBA)
            self.updateFromParent()

    def chooseGridColorOther(self):
        newColor = QtGui.QColorDialog.getColor()
        if newColor.isValid():
            # user selected a new color
            newColorRGB = newColor.getRgbF()
            curAlpha = self.gridOpacityOtherSlider.value() / 100.0
            newColorRGBA = newColorRGB[0:3]+(curAlpha,)
            self.displayWindow.setGrid(gridColorOther=newColorRGBA)
            self.updateFromParent()

    def opacityChanged(self):
        curColor = self.gridColorPalette.color(QtGui.QPalette.Base)
        curColorRGB = curColor.getRgbF()[0:3]
        newAlpha = self.gridOpacitySlider.value() / 100.0
        self.gridOpacityDisplay.setText('Opacity: %d%%' % self.gridOpacitySlider.value())
        newColorRGBA = curColorRGB[0:3]+(newAlpha,)
        self.displayWindow.setGrid(gridColor=newColorRGBA)
        self.updateFromParent()

    def opacityOtherChanged(self):
        curColor = self.gridColorOtherPalette.color(QtGui.QPalette.Base)
        curColorRGB = curColor.getRgbF()[0:3]
        newAlpha = self.gridOpacityOtherSlider.value() / 100.0
        self.gridOpacityOtherDisplay.setText('Opacity: %d%%' % self.gridOpacityOtherSlider.value())
        newColorRGBA = curColorRGB[0:3]+(newAlpha,)
        self.displayWindow.setGrid(gridColorOther=newColorRGBA)
        self.updateFromParent()


class LensletSettings(QtGui.QWidget):
    """
    A window that has settings on how the image is multiplexed
    """
    def __init__(self, displayWindow, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.updating = False
        
        self.displayWindow = displayWindow

        self.centerGroup = QtGui.QGroupBox('Center lenslet position', self)
        self.offsetXSpinBox = QtGui.QDoubleSpinBox()
        self.offsetXSpinBox.setRange(-20000.0,20000.0)
        self.offsetXSpinBox.setDecimals(3)
        self.offsetXSpinBox.setValue(256.5)
        self.offsetXSpinBox.setSingleStep(0.1)
        self.offsetXSpinBox.setSuffix(' pixel(s)')
        self.offsetYSpinBox = QtGui.QDoubleSpinBox()
        self.offsetYSpinBox.setRange(-20000.0,20000.0)
        self.offsetYSpinBox.setDecimals(3)
        self.offsetYSpinBox.setValue(256.5)
        self.offsetYSpinBox.setSingleStep(0.1)
        self.offsetYSpinBox.setSuffix(' pixel(s)')
        self.centerLayout = QtGui.QGridLayout(self.centerGroup)
        self.centerLayout.addWidget(QtGui.QLabel('x:'),0,0)
        self.centerLayout.addWidget(self.offsetXSpinBox,0,1)
        self.centerLayout.addWidget(QtGui.QLabel('y:'),0,3)
        self.centerLayout.addWidget(self.offsetYSpinBox,0,4)
        self.centerLayout.setColumnStretch(2,1)
        self.centerLayout.setSpacing(0)
        self.centerGroup.setLayout(self.centerLayout)

        self.horizGroup = QtGui.QGroupBox('One lenslet to the right', self)
        self.horizXSpinBox = QtGui.QDoubleSpinBox()
        self.horizXSpinBox.setDecimals(3)
        self.horizXSpinBox.setValue(17.0)
        self.horizXSpinBox.setMinimum(-1000.0)
        self.horizXSpinBox.setMaximum(1000.0)
        self.horizXSpinBox.setSingleStep(0.005)
        self.horizXSpinBox.setSuffix(' pixel(s)')
        self.horizYSpinBox = QtGui.QDoubleSpinBox()
        self.horizYSpinBox.setDecimals(3)
        self.horizYSpinBox.setValue(0.0)
        self.horizYSpinBox.setMinimum(-1000.0)
        self.horizYSpinBox.setMaximum(1000.0)
        self.horizYSpinBox.setSingleStep(0.005)
        self.horizYSpinBox.setSuffix(' pixel(s)')
        self.horizLayout = QtGui.QGridLayout(self.horizGroup)
        self.horizLayout.addWidget(QtGui.QLabel('dx:'),0,0)
        self.horizLayout.addWidget(self.horizXSpinBox,0,1)
        self.horizLayout.addWidget(QtGui.QLabel('dy:'),0,3)
        self.horizLayout.addWidget(self.horizYSpinBox,0,4)
        self.horizLayout.setColumnStretch(2,1)
        self.horizLayout.setSpacing(0)
        self.horizGroup.setLayout(self.horizLayout)

        self.vertGroup = QtGui.QGroupBox('One lenslet down', self)
        self.vertXSpinBox = QtGui.QDoubleSpinBox()
        self.vertXSpinBox.setDecimals(3)
        self.vertXSpinBox.setValue(0.0)
        self.vertXSpinBox.setMinimum(-1000.0)
        self.vertXSpinBox.setMaximum(1000.0)
        self.vertXSpinBox.setSingleStep(0.005)
        self.vertXSpinBox.setSuffix(' pixel(s)')
        self.vertYSpinBox = QtGui.QDoubleSpinBox()
        self.vertYSpinBox.setDecimals(3)
        self.vertYSpinBox.setValue(17.0)
        self.vertYSpinBox.setMinimum(-1000.0)
        self.vertYSpinBox.setMaximum(1000.0)
        self.vertYSpinBox.setSingleStep(0.005)
        self.vertYSpinBox.setSuffix(' pixel(s)')
        self.vertLayout = QtGui.QGridLayout(self.vertGroup)
        self.vertLayout.addWidget(QtGui.QLabel('dx:'),0,0)
        self.vertLayout.addWidget(self.vertXSpinBox,0,1)
        self.vertLayout.addWidget(QtGui.QLabel('dy:'),0,3)
        self.vertLayout.addWidget(self.vertYSpinBox,0,4)
        self.vertLayout.setSpacing(0)
        self.vertLayout.setColumnStretch(2,1)
        self.vertGroup.setLayout(self.vertLayout)

        spinboxToolTip = 'Enter a value, use the up/down arrows, or use mouse scroll.\nHold down Ctrl (Command on Mac) to scroll faster using the mouse.'
        self.offsetXSpinBox.setToolTip(spinboxToolTip)
        self.offsetYSpinBox.setToolTip(spinboxToolTip)
        self.horizXSpinBox.setToolTip(spinboxToolTip)
        self.horizYSpinBox.setToolTip(spinboxToolTip)
        self.vertXSpinBox.setToolTip(spinboxToolTip)
        self.vertYSpinBox.setToolTip(spinboxToolTip)

        self.buttonGroup = QtGui.QGroupBox('', self)
        self.loadButton = QtGui.QPushButton('&Load')
        self.loadButton.setToolTip('Load the lenslet parameters from a file')
        self.saveButton = QtGui.QPushButton('&Save')
        self.saveButton.setToolTip('Save the lenslet parameters to a file')
        self.recenterButton = QtGui.QPushButton('Re&center')
        self.recenterButton.setToolTip('Reset center lenslet position to default settings')
        self.resetButton = QtGui.QPushButton('&Reset')
        self.resetButton.setToolTip('Reset center lenslet position and lenslet spacing to default settings')
        self.flipVButton = QtGui.QPushButton('&Flip vertically')
        self.flipVButton.setToolTip('Change the lenslet parameters so that they correspond to a vertically-flipped image')
        self.rotate180Button = QtGui.QPushButton('R&otate 180')
        self.rotate180Button.setToolTip('Change the lenslet parameters so that they correspond to a 180-degree rotated image')
        self.showLensletsButton = QtGui.QPushButton('S&how rectification')
        self.showLensletsButton.setToolTip("Display the raw light field image as well as lenslet grid for rectification")

        self.shiftCenterButton = QtGui.QPushButton('Sh&ift center')
        self.shiftCenterButton.setToolTip("Shift the center lenslet position to as close to the center of the image as possible while still retaining the same grid")
        self.loadWarpButton = QtGui.QPushButton('Load warp')
        self.loadWarpButton.setToolTip('Load from ImageStack -lfrectify warp settings')
        self.saveWarpButton = QtGui.QPushButton('Save warp')
        self.saveWarpButton.setToolTip('Save to ImageStack -lfrectify warp settings')
        # no more ImageStack support
        self.loadWarpButton.setVisible(False)
        self.saveWarpButton.setVisible(False)
        
        self.buttonLayout = QtGui.QGridLayout(self.buttonGroup)
        self.buttonLayout.addWidget(self.loadButton,0,0)
        self.buttonLayout.addWidget(self.saveButton,0,1)
        self.buttonLayout.addWidget(self.loadWarpButton,1,0)
        self.buttonLayout.addWidget(self.saveWarpButton,1,1)
        self.buttonLayout.addWidget(self.flipVButton,2,0)
        self.buttonLayout.addWidget(self.rotate180Button,2,1)
        self.buttonLayout.addWidget(self.recenterButton,3,0)
        self.buttonLayout.addWidget(self.resetButton,3,1)
        self.buttonLayout.addWidget(self.shiftCenterButton,3,2)
        self.buttonLayout.addWidget(self.showLensletsButton,4,0)
        self.buttonLayout.setColumnStretch(3,1)
        self.buttonLayout.setRowStretch(5,1)
        self.buttonGroup.setLayout(self.buttonLayout)

        rectificationDirections = ["Click 'Show rectification' to display a grid overlay over the raw light field image.",
                                   "Click 'Recenter' if grid roughly matches lenslet images, or 'Reset' if grid is way off.",
                                   "Scroll the raw image by dragging the right mouse button (Control-drag on Mac also works) over the image until you can see the center lenslet 'box' (this is brighter than the other boxes and generally will contain a circle)",
                                   "Zoom in and out by using the mouse scroll wheel over the image or by using Ctrl with the +/- buttons (Command on Mac).",
                                   "Find the center of the closest lenslet image and adjust the center lenslet position until the center of the circle matches the center of the closest lenslet image.",
                                   "Adjust the 'One lenslet to the right' values so that the box immediately to the right of the center box roughly matches that of the lenslet immediately to the right.",
                                   "Scroll the image to the right edge and follow the horizontal lenslet grid pattern, making adjustments to 'One lenslet to the right' as necessary.",
                                   "Repeat for 'One lenslet down'.",
                                   "Save the lenslet rectification settings using 'Save lenslets'"]
        directionsText = '<br>'.join([str(i+1)+'. '+x for (x,i) in zip(rectificationDirections,range(len(rectificationDirections)))])

        self.directions = QtGui.QGroupBox('How to rectify')
        self.directionsBox = QtGui.QTextEdit(directionsText)
        self.directionsBox.setReadOnly(True)
        self.directionsLayout = QtGui.QGridLayout(self.directions)
        self.directionsLayout.addWidget(self.directionsBox,0,0)
        self.directions.setLayout(self.directionsLayout)

        self.settingsLayout = QtGui.QGridLayout(self)
        self.settingsLayout.addWidget(self.centerGroup,0,0)
        self.settingsLayout.addWidget(self.horizGroup,1,0)
        self.settingsLayout.addWidget(self.vertGroup,2,0)
        self.settingsLayout.addWidget(self.buttonGroup,3,0)
        self.settingsLayout.addWidget(self.directions,4,0)
        self.settingsLayout.setRowStretch(4,1)
        self.setLayout(self.settingsLayout)

        self.updateFromParent()

        self.connect(self.offsetXSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.offsetChanged)
        self.connect(self.offsetYSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.offsetChanged)

        self.connect(self.horizXSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.horizChanged)
        self.connect(self.horizYSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.horizChanged)
        self.connect(self.vertXSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.vertChanged)
        self.connect(self.vertYSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.vertChanged)

        self.connect(self.loadButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.load)
        self.connect(self.saveButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.save)
        self.connect(self.recenterButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.recenter)
        self.connect(self.resetButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.reset)
        self.connect(self.flipVButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.flipVertical)
        self.connect(self.rotate180Button,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.rotate180)
        self.connect(self.showLensletsButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.showLenslets)

        self.connect(self.shiftCenterButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.shiftCenter)
        self.connect(self.loadWarpButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.loadWarp)
        self.connect(self.saveWarpButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.saveWarp)

        self.connect(self.displayWindow, QtCore.SIGNAL('lensletChanged()'),
                     self.updateFromParent)

    def updateFromParent(self):
        self.updating = True
        # update offset from parent
        self.offsetXSpinBox.setValue(self.displayWindow.lensletOffset[0])
        self.offsetYSpinBox.setValue(self.displayWindow.lensletOffset[1])
        # update the horizontal and vertical basis vectors
        self.horizXSpinBox.setValue(self.displayWindow.lensletHoriz[0])
        self.horizYSpinBox.setValue(self.displayWindow.lensletHoriz[1])
        self.vertXSpinBox.setValue(self.displayWindow.lensletVert[0])
        self.vertYSpinBox.setValue(self.displayWindow.lensletVert[1])
        self.updating = False

    def showLenslets(self):
        self.displayWindow.emitDisplayModeChanged(0)
        self.displayWindow.displaySettings.gridBoundaries.click()

    def getLensletParams(self):
        "Get all the lenslet parameters"
        return [self.offsetXSpinBox.value(),
                self.offsetYSpinBox.value(),
                self.horizXSpinBox.value(),
                self.horizYSpinBox.value(),
                self.vertXSpinBox.value(),
                self.vertYSpinBox.value()]

    def setLensletParams(self, params):
        "Set the lenslet parameters all at once"
        self.offsetXSpinBox.setValue(params[0])
        self.offsetYSpinBox.setValue(params[1])
        self.horizXSpinBox.setValue(params[2])
        self.horizYSpinBox.setValue(params[3])
        self.vertXSpinBox.setValue(params[4])
        self.vertYSpinBox.setValue(params[5])

    def warpToLenslet(self, (x, y, dx, dy, sx, sy)):
        "Convert warp parameters to lenslet parameters"
        invDet = 1.0 / ( 1 + dx*dy)
        rightX = sx * invDet
        downX = sy*dx * invDet
        rightY = sx*dy * invDet
        downY = sy * invDet
        offsetX = (x + dx*y) * invDet + 0.5*(downX+rightX) - 0.5
        offsetY = (y + dy*x) * invDet + 0.5*(downY+rightY) - 0.5
        return (offsetX, offsetY, rightX, rightY, downX, downY)

    def lensletToWarp(self, (offsetX, offsetY, rightX, rightY, downX, downY)):
        "Convert lenslet parameters to warp parameters"
        dy = rightY/rightX
        dx = downX/downY
        det = (1+dx*dy)
        sx = rightX * det
        sy = downY * det
        tempx = (offsetX-0.5*(downX+rightX) + 0.5)*det
        tempy = (offsetY-0.5*(downY+rightY) + 0.5)*det
        invDet = 1.0/(1-dx*dy)
        x = (tempx - dx*tempy) * invDet
        y = (-dy*tempx + tempy) * invDet
        return (x, y, dx, dy, sx, sy)

    def shiftCenter(self):
        "Shift the center to as close to the center of the image as possible while still retaining the same grid"
        params = self.getLensletParams()
        width, height = self.displayWindow.textureSize
        # get center of image
        centerX = (width-1)*0.5
        centerY = (height-1)*0.5
        offsetX = centerX - params[0]
        offsetY = centerY - params[1]
        # apply inverse rotation/skew matrix
        invDet = 1.0/(params[2]*params[5] - params[3]*params[4])
        lensletX = invDet*(offsetX*params[5] - offsetY*params[4])
        lensletY = invDet*(offsetY*params[2] - offsetX*params[3])
        # round to nearest lenslet
        lensletX = round(lensletX)
        lensletY = round(lensletY)
        # apply forward matrix
        offsetX = params[2]*lensletX + params[4]*lensletY
        offsetY = params[3]*lensletX + params[5]*lensletY
        # change settings
        params[0] = offsetX + params[0]
        params[1] = offsetY + params[1]
        self.setLensletParams(params)

    def flipVertical(self):
        "Change the lenslet settings such that the original settings corresponded to a vertically flipped version of the image"
        params = self.getLensletParams()
        width, height = self.displayWindow.textureSize
        # set the centering value
        params[1] = height-1-params[1]
        # flip the sign for basis vectors
        params[3] = -params[3]
        params[4] = -params[4]
        self.setLensletParams(params)

    def rotate180(self):
        "Change the lenslet settings such that the original settings corresponded to a 180 degree rotated version of the image"
        params = self.getLensletParams()
        width, height = self.displayWindow.textureSize
        # set the centering value
        params[0] = width-1-params[0]
        params[1] = height-1-params[1]
        self.setLensletParams(params)

    def load(self):
        "Load values from a file"
        lenslet_path = self.displayWindow.settings.getString('input/file/default_path','')
        filepath = QtGui.QFileDialog.getOpenFileName(self.displayWindow,
                                                     'Choose a text file with lenslet settings',
                                                     lenslet_path,
                                                     'Lenslet parameter files (*.txt);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.displayWindow.settings.setValue('input/file/default_path',new_path)
            try:
                f=open(filepath,'r')
                line=f.readline()
                f.close()
                line = line.strip()
                values2 = eval(line)
                values = [values2[i] for i in range(6)]
                self.offsetXSpinBox.setValue(values[0])
                self.offsetYSpinBox.setValue(values[1])
                self.horizXSpinBox.setValue(values[2])
                self.horizYSpinBox.setValue(values[3])
                self.vertXSpinBox.setValue(values[4])
                self.vertYSpinBox.setValue(values[5])
            except Exception, e:
                raise Error('Unable to parse lenslet settings file')

    def loadWarp(self):
        "Load values from a ImageStack lfrectify warp file"
        lenslet_path = self.displayWindow.settings.getString('input/file/default_path','')
        filepath = QtGui.QFileDialog.getOpenFileName(self.displayWindow,
                                                     'Choose a text file with ImageStack -lfrectify warp parameters',
                                                     lenslet_path,
                                                     'ImageStack lfrectify parameter files (*.warp);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.displayWindow.settings.setValue('input/file/default_path',new_path)
            try:
                f=open(filepath,'r')
                lineTokens = [x.split() for x in f.readlines()]
                # make sure we have a correct list of parameters
                paramDict = {}
                for i in range(6):
                    assert(len(lineTokens[i]) == 3)
                    paramDict[lineTokens[i][0]+'_'+lineTokens[i][1]] = float(lineTokens[i][2])
                warpParams = (paramDict['translate_x'],
                              paramDict['translate_y'],
                              paramDict['shear_x'],
                              paramDict['shear_y'],
                              paramDict['scale_x'],
                              paramDict['scale_y'])
                params = self.warpToLenslet(warpParams)
                self.setLensletParams(params)
            except Exception, e:
                raise Error('Unable to parse ImageStack lfrectify settings file')

    def save(self):
        "Save values to a file"
        lenslet_path = self.displayWindow.settings.getString('input/file/default_path')
        filepath = QtGui.QFileDialog.getSaveFileName(self.displayWindow,
                                                     'Please choose a text file where lenslet settings will be saved',
                                                     lenslet_path,
                                                     'Lenslet parameter files (*.txt);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.displayWindow.settings.setValue('input/file/default_path',new_path)
            f=open(filepath,'w')
            f.write('(%f,%f,%f,%f,%f,%f)\n' % (self.offsetXSpinBox.value(),
                                               self.offsetYSpinBox.value(),
                                               self.horizXSpinBox.value(),
                                               self.horizYSpinBox.value(),
                                               self.vertXSpinBox.value(),
                                               self.vertYSpinBox.value()))
            f.write('# (x-offset,y-offset,right-dx,right-dy,down-dx,down-dy)')
            f.close()
            QtGui.QMessageBox.information(self.displayWindow,
                                          'Lenslet settings saved',
                                          'Lenslet settings have been saved to %s' % filepath)
                                        
    def saveWarp(self):
        "Save ImageStack lfrectify warp parameters to a file"
        lenslet_path = self.displayWindow.settings.getString('input/file/default_path')
        filepath = QtGui.QFileDialog.getSaveFileName(self.displayWindow,
                                                     'Please choose a file where ImageStack -lfrectify parameters will be saved',
                                                     lenslet_path,
                                                     'ImageStack lfrectify parameter files (*.warp);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.displayWindow.settings.setValue('input/file/default_path',new_path)
            f=open(filepath,'w')
            # get the current parameters
            params = self.getLensletParams()
            # convert to warp parameters
            warpParams = self.lensletToWarp(params)
            # write it out
            f.write('translate x %lf\ntranslate y %lf\nshear x %lf\nshear y %lf \nscale x %lf\nscale y %lf\ndesired lenslet size 20\n' % warpParams)
            f.close()
            QtGui.QMessageBox.information(self.displayWindow,
                                          'ImageStack lfrectify parameters saved',
                                          'ImageStack lfrectify parameters have been saved to %s' % filepath)

    def reset(self):
        "Reset to default values"
        textureSize = self.displayWindow.textureSize
        self.displayWindow.setGrid(lensletOffset=((textureSize[0]-1)*0.5,
                                                  (textureSize[1]-1)*0.5),
                                   lensletHoriz=(17.0,0.0),
                                   lensletVert=(0.0,17.0))
        self.updateFromParent()

    def recenter(self):
        "Move the grid back to the center"
        textureSize = self.displayWindow.textureSize
        self.displayWindow.setGrid(lensletOffset=((textureSize[0]-1)*0.5,
                                                  (textureSize[1]-1)*0.5))
        self.updateFromParent()

    def offsetChanged(self):
        if self.updating:
            return
        newOffset = (self.offsetXSpinBox.value(),
                     self.offsetYSpinBox.value())
        self.displayWindow.setGrid(lensletOffset=newOffset)
        self.updateFromParent()

    def horizChanged(self):
        if self.updating:
            return
        newHoriz = (self.horizXSpinBox.value(),
                    self.horizYSpinBox.value())
        self.displayWindow.setGrid(lensletHoriz=newHoriz)
        self.updateFromParent()

    def vertChanged(self):
        if self.updating:
            return
        newVert = (self.vertXSpinBox.value(),
                   self.vertYSpinBox.value())
        self.displayWindow.setGrid(lensletVert=newVert)
        self.updateFromParent()
        
class OpticsSettings(QtGui.QWidget):
    """
    A simple way to manipulate the ray transfer matrix
    """
    def __init__(self, displayWindow, parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        self.displayWindow = displayWindow

        self.focusDirections='\nMoving z in the positive direction (dragging/pressing right) brings the virtual focal plane closer to the objective.'

        self.focusGroup = gui.TwinInfiniteSliderWidget(5.0, 0.5, 10,
                                                       label='Focus (z)',
                                                       suffix='um',
                                                       digits=4,
                                                       eps=1e-6,
                                                       directions=self.focusDirections)
        self.connect(self.focusGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.rayTransferChanged)

        # NOT DISPLAYED
        self.perspectiveGroup = gui.SliderWidget(gui.LinearMap(-1.0,1.0),
                                                 gui.FloatDisplay(3),
                                                 0.0,
                                                 'Perspective')
        self.connect(self.perspectiveGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.rayTransferChanged)



        self.apertureTypeGroup = QtGui.QGroupBox('Aperture diameter', self)
        self.pinholeType = QtGui.QRadioButton('&Pinhole')
        self.variableType = QtGui.QRadioButton('&Custom')
        self.fullType = QtGui.QRadioButton('&Full')
        self.connect(self.pinholeType, QtCore.SIGNAL('clicked()'),
                     self.apertureChanged)
        self.connect(self.variableType, QtCore.SIGNAL('clicked()'),
                     self.apertureChanged)
        self.connect(self.fullType, QtCore.SIGNAL('clicked()'),
                     self.apertureChanged)
        self.apertureTypeLayout = QtGui.QGridLayout(self.apertureTypeGroup)
        self.apertureTypeLayout.addWidget(self.pinholeType, 0, 0)
        self.apertureTypeLayout.addWidget(self.variableType, 0, 1)
        self.apertureTypeLayout.addWidget(self.fullType, 0, 2)
        self.apertureTypeGroup.setLayout(self.apertureTypeLayout)

        self.apertureGroup = gui.SliderWidget(gui.LinearMap(0.0,1.0),
                                              gui.PercentDisplay(1),
                                              0.5,
                                              'Custom effective aperture',
                                              suffix='%',
                                              steps=1000)
        self.connect(self.apertureGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.apertureChanged)

        self.pinholeType.click()

        self.apertureSamplesGroup = gui.SliderWidget(gui.LinearIntMap(1,APWIDTH*APWIDTH),
                                                     (int,str),
                                                     int(APWIDTH*APWIDTH/2),
                                                     'Number of aperture samples for non-pinhole rendering',
                                                     steps=APWIDTH*APWIDTH)

        self.normalizedAperture = aperture.getCircularAperture(self.apertureSamplesGroup.value())
        self.connect(self.apertureSamplesGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.samplesChanged)

        self.buttonGroup = QtGui.QGroupBox('', self)
        self.centerButton = QtGui.QPushButton('&Reset pan')
        self.centerButton.setToolTip('Recenter the panning')
        self.resetFocusButton = QtGui.QPushButton('Reset &focus')
        self.resetFocusButton.setToolTip('Set the focus back to 0.0')
        self.buttonLayout = QtGui.QGridLayout(self.buttonGroup)
        self.buttonLayout.addWidget(self.centerButton,0,0)
        self.buttonLayout.addWidget(self.resetFocusButton,0,1)
        self.buttonLayout.setColumnStretch(2,1)
        self.buttonLayout.setRowStretch(1,1)
        self.buttonGroup.setLayout(self.buttonLayout)

        self.recipeGroup = QtGui.QGroupBox('Optics Recipe', self)
        recipe = self.displayWindow.recipe()
        self.pitchSelector = gui.CustomOptionSelectorWidget('Microlens Array Pitch:',
                                                            [('62.5um', 62.5),
                                                             ('125um', 125.0),
                                                             ('250um', 250.0)],
                                                            'Custom', float, self)
        self.pitchSelector.setValue(recipe['pitch'])
        self.flenSelector = gui.CustomOptionSelectorWidget('Microlens Focal Length:',
                                                           [('1600um', 1600.0),
                                                            ('2500um', 2500.0),
                                                            ('3750um', 3750.0)],
                                                            'Custom', float, self)
        self.flenSelector.setValue(recipe['flen'])
        self.magSelector = gui.CustomOptionSelectorWidget('Objective Magnification:',
                                                          [('10X',10.0),
                                                           ('20X',20.0),
                                                           ('40X',40.0),
                                                           ('60X',60.0),
                                                           ('63X',63.0),
                                                           ('100X',100.0)],
                                                          'Custom', float,
                                                          self)
        self.magSelector.setValue(recipe['mag'])
        self.abbeSelector = QtGui.QGridLayout()
        self.abbeCheckBox = QtGui.QCheckBox('Paraxial approximation', self)
        self.abbeCheckBox.setChecked(not recipe['abbe'])
        self.abbeSelector.addWidget(self.abbeCheckBox, 0, 1)
        self.abbeSelector.setColumnStretch(0,1)
        self.naSelector = gui.CustomOptionSelectorWidget('Objective NA:',
                                                         [('0.45',0.45),
                                                          ('0.8',0.8),
                                                          ('0.95',0.95),
                                                          ('1.0',1.0),
                                                          ('1.3',1.3)],
                                                         'Custom', float,
                                                         self)
        self.naSelector.setValue(recipe['na'])
        (minLabel,maxLabel) = self.apertureGroup.rangeLabels(defaults=True)
        self.apertureGroup.setRangeLabels(None, maxLabel+'(%g NA)' % recipe['na'])
        self.mediumSelector = gui.CustomOptionSelectorWidget('Medium Refractive Index:',
                                                             [('Dry (air)',1.0),
                                                              ('Water',1.333),
                                                              ('Oil',1.5),
                                                              ],
                                                             'Custom', float,
                                                             self)
        self.mediumSelector.setValue(recipe['medium'])
        self.recipeLayout = QtGui.QGridLayout(self.recipeGroup)
        num_items = 0
        self.recipeLayout.addWidget(self.pitchSelector,num_items,0)
        self.connect(self.pitchSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.recipeChanged)
        num_items += 1 
        self.recipeLayout.addWidget(self.flenSelector,num_items,0)
        self.connect(self.flenSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.recipeChanged)
        num_items += 1 
        self.recipeLayout.addWidget(self.magSelector,num_items,0)
        self.connect(self.magSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.recipeChanged)
        num_items += 1 
        self.recipeLayout.addLayout(self.abbeSelector,num_items,0)
        self.connect(self.abbeCheckBox,
                     QtCore.SIGNAL('stateChanged(int)'),
                     self.recipeChanged)
        num_items += 1 
        self.recipeLayout.addWidget(self.naSelector,num_items,0)
        self.connect(self.naSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.recipeChanged)
        self.connect(self.naSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.naChanged)
        num_items += 1 
        self.recipeLayout.addWidget(self.mediumSelector,num_items,0)
        self.connect(self.mediumSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.recipeChanged)
        num_items += 1

        self.recipeButtons = QtGui.QGridLayout()
        self.recipeNote = QtGui.QLabel('')
        if recipe['na'] >= recipe['medium']:
            self.recipeNote.setText('<font color="red">Error: NA >= medium index</font>')
        self.loadRecipeButton = QtGui.QPushButton('Load')
        self.saveRecipeButton = QtGui.QPushButton('Save')
        self.recipeButtons.addWidget(self.recipeNote, 0, 0)
        self.recipeButtons.addWidget(self.loadRecipeButton, 0, 1)
        self.recipeButtons.addWidget(self.saveRecipeButton, 0, 2)
        self.recipeButtons.setColumnStretch(0,1)

        self.recipeLayout.addLayout(self.recipeButtons,num_items,0)
        num_items += 1
        
        self.recipeLayout.setColumnStretch(1,1)
        self.recipeLayout.setRowStretch(num_items,1)
        self.recipeGroup.setLayout(self.recipeLayout)
        
        self.settingsLayout = QtGui.QGridLayout(self)
        self.settingsLayout.addWidget(self.focusGroup,0,0)
        self.settingsLayout.addWidget(self.perspectiveGroup,1,0)
        self.settingsLayout.addWidget(self.apertureTypeGroup,2,0)
        self.settingsLayout.addWidget(self.apertureGroup,3,0)
        self.settingsLayout.addWidget(self.apertureSamplesGroup,4,0)
        self.settingsLayout.addWidget(self.buttonGroup,5,0)
        self.settingsLayout.addWidget(self.recipeGroup,6,0)

        if not ENABLE_PERSPECTIVE:
            self.perspectiveGroup.setVisible(False)
        
        self.settingsLayout.setRowStretch(7,1)
        self.setLayout(self.settingsLayout)

        self.connect(self.centerButton,
                     QtCore.SIGNAL('clicked()'),
                     self.displayWindow.setUV)
        self.connect(self.resetFocusButton,
                     QtCore.SIGNAL('clicked()'),
                     self.resetFocus)

        self.connect(self.loadRecipeButton,
                     QtCore.SIGNAL('clicked()'),
                     self.loadRecipe)
        self.connect(self.saveRecipeButton,
                     QtCore.SIGNAL('clicked()'),
                     self.saveRecipe)

        self.connect(self.displayWindow,
                     QtCore.SIGNAL('shaderChanged(int)'),
                     self.updateFocus)

    def rayTransferChanged(self):
        """
        When the user selects a different focal plane or perspective
        """
        focus = self.focusGroup.value()
        perspective = self.perspectiveGroup.value()
        RTM = [[1.0-focus*perspective, 0.0, -focus, 0.0],
               [0.0, 1.0-focus*perspective, 0.0, -focus],
               [perspective, 0.0, 1.0, 0.0],
               [0.0, perspective, 0.0, 1.0]]
        RTM = numpy.array(RTM,dtype='float32')
        self.displayWindow.setMatrix(RTM=RTM)

    def processDisplayModeChanged(self, num):
        """
        Process a display mode change
        """
        # disable/enable focus slider
        if self.pinholeType.isChecked() or num != 2:
            self.focusGroup.setEnabled(False)
            self.focusGroup.setToolTip('Set aperture type to be something other than pinhole,\nand make sure display mode is in 3D (light field)\nto adjust focus in light field rendering')
        else:
            self.focusGroup.setEnabled(True)
            self.focusGroup.setToolTip('')
        self.apertureGroup.setEnabled(self.variableType.isChecked())
        if num == 1:
            # reset the focus and pan
            self.centerButton.click()
            self.resetFocusButton.click()
            # select pinhole
            self.pinholeType.setChecked(True)
            self.apertureChanged()
            
    def updateFocus(self):
        """
        Update the state of the focus widget based on the
        current conditions
        """
        if self.pinholeType.isChecked() or self.displayWindow.shaderNext != 2:
            self.focusGroup.setEnabled(False)
            self.focusGroup.setToolTip('Set aperture type to be something other than pinhole,\nand make sure display mode is in 3D (light field)\nto adjust focus in light field rendering')
        else:
            self.focusGroup.setEnabled(True)
            self.focusGroup.setToolTip('')
        self.apertureGroup.setEnabled(self.variableType.isChecked())

    def apertureChanged(self):
        """
        When the user selects a different aperture size
        """
        self.updateFocus()
        if self.pinholeType.isChecked():
            aperture = [(0.0,0.0,1.0)]
        else:
            # tell the view mode toolbar that we need to go to full 3D mode
            self.emit(QtCore.SIGNAL('displayModeChanged(int)'), 2)
            if self.variableType.isChecked():
                apertureDiameter = self.apertureGroup.value()
            else:
                apertureDiameter = 1.0
            apertureScale = apertureDiameter * self.displayWindow.maxNormalizedSlope() / 0.5
            aperture = [(x*apertureScale,y*apertureScale,w) for (x,y,w) in self.normalizedAperture]
            if not aperture:
                aperture = [(0.0,0.0,1.0)]
        self.displayWindow.aperture = aperture
        self.displayWindow.apertureDirty =  True
        self.displayWindow.dirty = True

    def recipeChanged(self):
        # check for valid numbers
        errors = []
        try:
            pitch=self.pitchSelector.value()
        except ValueError:
            errors.append('pitch')
        try:
            flen=self.flenSelector.value()
        except ValueError:
            errors.append('focal length')
        try:
            mag=self.magSelector.value()
        except ValueError:
            errors.append('magnification')
        try:
            na=self.naSelector.value()
        except ValueError:
            errors.append('NA')
        try:
            medium=self.mediumSelector.value()
        except ValueError:
            errors.append('medium index')
        if errors:
            self.recipeNote.setText('<font color="red">Error: invalid '+', '.join(errors)+'</font>')
        elif na >= medium:
            self.recipeNote.setText('<font color="red">Error: NA >= medium index</font>')
        else:
            self.recipeNote.setText('')
            self.displayWindow.setRecipe(pitch=pitch,
                                         flen=flen,
                                         mag=mag,
                                         abbe=not self.abbeCheckBox.isChecked(),
                                         na=na,
                                         medium=medium)

    def samplesChanged(self):
        self.normalizedAperture = aperture.getCircularAperture(self.apertureSamplesGroup.value())
        self.apertureChanged()
        
    def resetFocus(self):
        self.focusGroup.setValue(0.0)

    def naChanged(self):
        "If the NA has been changed, update effective aperture slider"
        (minLabel,maxLabel) = self.apertureGroup.rangeLabels(defaults=True)
        self.apertureGroup.setRangeLabels(None, maxLabel+'(%g NA)' % self.naSelector.value())        

    def loadRecipe(self):
        "Load optics recipe from a file"
        lenslet_path = self.displayWindow.settings.getString('input/file/default_path','')
        filepath = QtGui.QFileDialog.getOpenFileName(self.displayWindow,
                                                     'Choose a configuration file for optics recipe',
                                                     lenslet_path,
                                                     'Text files (*.txt);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.displayWindow.settings.setValue('input/file/default_path',new_path)
            try:
                f=open(filepath,'r')
                lineTokens = [x.split() for x in f.readlines()]
                # make sure we have a correct list of parameters
                paramDict = {}
                for i in range(6):
                    assert(len(lineTokens[i]) == 2)
                    paramDict[lineTokens[i][0]] = lineTokens[i][1]
                params = {'pitch':float(paramDict['pitch']),
                          'flen':float(paramDict['flen']),
                          'mag':float(paramDict['mag']),
                          'abbe':(paramDict['abbe'].lower() in ['true','yes','y']),
                          'na':float(paramDict['na']),
                          'medium':float(paramDict['medium'])}
                self.displayWindow.setRecipe(**params)
            except Exception, e:
                QtGui.QMessageBox.critical(self.displayWindow,
                                           'Error',
                                           'Unable to parse optics recipe file')
                raise Error('Unable to parse optics recipe file')
            recipe = self.displayWindow.recipe()
            self.pitchSelector.setValue(recipe['pitch'])
            self.flenSelector.setValue(recipe['flen'])
            self.magSelector.setValue(recipe['mag'])
            self.abbeCheckBox.setChecked(not recipe['abbe'])
            self.naSelector.setValue(recipe['na'])
            self.mediumSelector.setValue(recipe['medium'])

    def saveRecipe(self):
        "Save optics recipe to a file"
        lenslet_path = self.displayWindow.settings.getString('input/file/default_path')
        filepath = QtGui.QFileDialog.getSaveFileName(self.displayWindow,
                                                     'Please choose a file where optics recipe will be saved',
                                                     lenslet_path,
                                                     'Text files (*.txt);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.displayWindow.settings.setValue('input/file/default_path',new_path)
            f=open(filepath,'w')
            # get the current parameters
            recipe = self.displayWindow.recipe()
            # write it out
            f.write('pitch %lg\n' % recipe['pitch'])
            f.write('flen %lg\n' % recipe['flen'])
            f.write('mag %lg\n' % recipe['mag'])
            f.write('abbe %s\n' % ['false','true'][recipe['abbe']])
            f.write('na %lg\n' % recipe['na'])
            f.write('medium %lg\n' % recipe['medium'])
            f.close()
            QtGui.QMessageBox.information(self.displayWindow,
                                          'Optics recipe saved',
                                          'Optics recipe has been saved to %s' % filepath)

        
