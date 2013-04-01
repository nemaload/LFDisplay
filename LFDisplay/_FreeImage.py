"""
Helper methods and data structures for interfacing with FreeImage
through ctypes
"""

import sys
import ctypes
import ctypes.util

class Error(Exception):
    pass

import os
import os.path

# dll loading helper for app bundles
def app_bundle_find_library(x):
    if 'RESOURCEPATH' in os.environ:
        _resource_paths = os.environ['RESOURCEPATH'].split(':')
        for path in _resource_paths:
            if os.path.exists(os.path.join(path,x)):
                return os.path.join(path,x)
    return ctypes.util.find_library(x)

# load the FreeImage dll

if sys.platform == 'win32':
    _dll = ctypes.WinDLL('FreeImage.dll')
    FUNCTYPE = ctypes.WINFUNCTYPE
elif sys.platform == 'darwin':
    _location = app_bundle_find_library('libfreeimage.3.dylib')
    if not _location:
        raise Error("Unable to locate FreeImage library")
    _dll = ctypes.CDLL(_location)
    FUNCTYPE = ctypes.CFUNCTYPE
else:
    _location = ctypes.util.find_library("freeimage")
    if not _location:
        raise Error("Unable to locate FreeImage library")
    _dll = ctypes.CDLL(_location)
    FUNCTYPE = ctypes.CFUNCTYPE

# initialize FreeImage

_dll.FreeImage_Initialise(0)

# general functions

_dll.FreeImage_GetVersion.restype = ctypes.c_char_p
_dll.FreeImage_GetVersion.argtypes = ()

def GetVersion():
    "Return a string containing the current version of the library"
    return _dll.FreeImage_GetVersion()

_dll.FreeImage_GetCopyrightMessage.restype = ctypes.c_char_p
_dll.FreeImage_GetCopyrightMessage.argtypes = ()

def GetCopyrightMessage():
    "Return a string containing a standard copyright message"
    return _dll.FreeImage_GetCopyrightMessage()

MESSAGEFUNC = FUNCTYPE(None,
                       ctypes.c_int, ctypes.c_char_p)

_dll.FreeImage_SetOutputMessage.restype = None
_dll.FreeImage_SetOutputMessage.argtypes = (MESSAGEFUNC,)

def SetOutputMessage(omf):
    "Set the output message function"
    _dll.FreeImage_SetOutputMessage(MESSAGEFUNC(omf))
    
# bitmap management functions

# image formats
FIF_UNKNOWN = -1
FIF_BMP	= 0
FIF_ICO	= 1
FIF_JPEG = 2
FIF_JNG	= 3
FIF_KOALA = 4
FIF_LBM	= 5
FIF_IFF = FIF_LBM
FIF_MNG	= 6
FIF_PBM	= 7
FIF_PBMRAW = 8
FIF_PCD	= 9
FIF_PCX	= 10
FIF_PGM	= 11
FIF_PGMRAW = 12
FIF_PNG	= 13
FIF_PPM	= 14
FIF_PPMRAW = 15
FIF_RAS	= 16
FIF_TARGA = 17
FIF_TIFF = 18
FIF_WBMP = 19
FIF_PSD	= 20
FIF_CUT	= 21
FIF_XBM	= 22
FIF_XPM	= 23
FIF_DDS	= 24
FIF_GIF = 25
FIF_HDR	= 26
FIF_FAXG3 = 27
FIF_SGI	= 28

# TIFF options
TIFF_DEFAULT = 0
TIFF_CMYK = 0x0001
TIFF_NONE = 0x0800 
TIFF_PACKBITS = 0x0100 
TIFF_DEFLATE = 0x0200 
TIFF_ADOBE_DEFLATE = 0x0400 
TIFF_CCITTFAX3 = 0x1000 
TIFF_CCITTFAX4 = 0x2000 
TIFF_LZW = 0x4000	
TIFF_JPEG = 0x8000

# Metadata models
[FIMD_COMMENTS,
 FIMD_EXIF_MAIN,
 FIMD_EXIF_EXIF,
 FIMD_EXIF_GPS,
 FIMD_EXIF_MAKERNOTE,
 FIMD_EXIF_INTEROP,
 FIMD_IPTC,
 FIMD_XMP,
 FIMD_GEOTIFF,
 FIMD_ANIMATION,
 FIMD_CUSTOM] = range(11)

# Metadata types
[FIDT_NOTYPE,
 FIDT_BYTE,
 FIDT_ASCII,
 FIDT_SHORT,
 FIDT_LONG,
 FIDT_RATIONAL,
 FIDT_SBYTE,
 FIDT_UNDEFINED,
 FIDT_SSHORT,
 FIDT_SLONG,
 FIDT_SRATIONAL,
 FIDT_FLOAT,
 FIDT_DOUBLE,
 FIDT_IFD,
 FIDT_PALETTE] = range(15)

# image types
FIT_UNKNOWN = 0 # unknown type
FIT_BITMAP = 1 # standard image			: 1-, 4-, 8-, 16-, 24-, 32-bit
FIT_UINT16 = 2 # array of unsigned short	: unsigned 16-bit
FIT_INT16 = 3 # array of short			: signed 16-bit
FIT_UINT32 = 4 # array of unsigned long	: unsigned 32-bit
FIT_INT32 = 5 # array of long			: signed 32-bit
FIT_FLOAT = 6 # array of float			: 32-bit IEEE floating point
FIT_DOUBLE = 7 # array of double			: 64-bit IEEE floating point
FIT_COMPLEX = 8 # array of FICOMPLEX		: 2 x 64-bit IEEE floating point
FIT_RGB16 = 9 # 48-bit RGB image			: 3 x 16-bit
FIT_RGBA16 = 10 # 64-bit RGBA image		: 4 x 16-bit
FIT_RGBF = 11 # 96-bit RGB float image	: 3 x 32-bit IEEE floating point
FIT_RGBAF = 12 # 128-bit RGBA float image	: 4 x 32-bit IEEE floating point

# color types
"""
Monochrome bitmap (1-bit) : first palette entry is white. Palletised
bitmap (4 or 8-bit) : the bitmap has an inverted greyscale palette
"""
FIC_MINISWHITE = 0
"""
Monochrome bitmap (1-bit) : first palette entry is black. Palletised
bitmap (4 or 8-bit) and single channel non standard bitmap: the
bitmap has a greyscale palette
"""
FIC_MINISBLACK = 1
"""
Palettized bitmap (1, 4 or 8 bit)
"""
FIC_PALETTE = 2
"""
High-color bitmap (16, 24 or 32 bit), RGB16 or RGBF
"""
FIC_RGB = 3
"""
High-color bitmap with an alpha channel (32 bit bitmap, RGBA16 or RGBAF)
"""
FIC_RGBALPHA = 4
"""
CMYK bitmap (32 bit only)
"""
FIC_CMYK = 5

# the image pointer
FIBITMAP = ctypes.c_void_p

_dll.FreeImage_AllocateT.restype = FIBITMAP
_dll.FreeImage_AllocateT.argtypes = (ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_uint,
                                     ctypes.c_uint,
                                     ctypes.c_uint)

def AllocateT(type, width, height, bpp=8, red_mask=0, green_mask=0, blue_mask=0):
    "Create a new FreeImage image"
    result = _dll.FreeImage_AllocateT(type,width,height,bpp,red_mask,green_mask,blue_mask)
    if not result:
        raise Error("Unable to allocate memory for an image")
    return result

_dll.FreeImage_Load.restype = FIBITMAP
_dll.FreeImage_Load.argtypes = (ctypes.c_int,
                                ctypes.c_char_p,
                                ctypes.c_int)
                                     
def Load(fif, filename, flags=0):
    "Load an image from a file"
    result = _dll.FreeImage_Load(fif, filename, flags)
    if not result:
        raise Error("Unable to load image file "+filename)
    return result

_dll.FreeImage_Save.restype = ctypes.c_int
_dll.FreeImage_Save.argtypes = (ctypes.c_int,
                                FIBITMAP,
                                ctypes.c_char_p,
                                ctypes.c_int)

def Save(fif, dib, filename, flags=0):
    "Save an image to a file"
    result = _dll.FreeImage_Save(fif, dib, filename, flags)
    if not result:
        raise Error("Unable to load image file "+filename)
    return result

_dll.FreeImage_Clone.restype = FIBITMAP
_dll.FreeImage_Clone.argtypes = (FIBITMAP,)

def Clone(dib):
    "Create a copy of an image"
    result = _dll.FreeImage_Clone(dib)
    if not result:
        raise Error("Unable to clone image")
    return result

_dll.FreeImage_Unload.restype = None
_dll.FreeImage_Unload.argtypes = (FIBITMAP,)

def Unload(dib):
    "Free an allocated image"
    _dll.FreeImage_Unload(dib)

# image information functions

_dll.FreeImage_GetImageType.restype = ctypes.c_int
_dll.FreeImage_GetImageType.argtypes = (FIBITMAP,)

def GetImageType(dib):
    return _dll.FreeImage_GetImageType(dib)

_dll.FreeImage_GetColorsUsed.restype = ctypes.c_int
_dll.FreeImage_GetColorsUsed.argtypes = (FIBITMAP,)

def GetColorsUsed(dib):
    return _dll.FreeImage_GetColorsUsed(dib)

_dll.FreeImage_GetBPP.restype = ctypes.c_int
_dll.FreeImage_GetBPP.argtypes = (FIBITMAP,)

def GetBPP(dib):
    return _dll.FreeImage_GetBPP(dib)

_dll.FreeImage_GetWidth.restype = ctypes.c_int
_dll.FreeImage_GetWidth.argtypes = (FIBITMAP,)

def GetWidth(dib):
    return _dll.FreeImage_GetWidth(dib)

_dll.FreeImage_GetHeight.restype = ctypes.c_int
_dll.FreeImage_GetHeight.argtypes = (FIBITMAP,)

def GetHeight(dib):
    return _dll.FreeImage_GetHeight(dib)

_dll.FreeImage_GetLine.restype = ctypes.c_int
_dll.FreeImage_GetLine.argyptes = (FIBITMAP,)

def GetLine(dib):
    return _dll.FreeImage_GetLine(dib)

_dll.FreeImage_GetPitch.restype = ctypes.c_int
_dll.FreeImage_GetPitch.argtypes = (FIBITMAP,)

def GetPitch(dib):
    return _dll.FreeImage_GetPitch(dib)

_dll.FreeImage_GetColorType.restype = ctypes.c_int
_dll.FreeImage_GetColorType.argtypes = (FIBITMAP,)

def GetColorType(dib):
    return _dll.FreeImage_GetColorType(dib)

_dll.FreeImage_GetPalette.restype = ctypes.c_void_p
_dll.FreeImage_GetPalette.argtypes = (FIBITMAP,)

def GetPalette(dib):
    bpp = _dll.FreeImage_GetBPP(dib)
    if bpp not in [1,4,8]:
        raise Error('Unsupported bit depth for palette')
    ptr = _dll.FreeImage_GetPalette(dib)
    return ctypes.string_at(ptr, 1<<(bpp+2))

def SetPalette(dib, pal):
    bpp = _dll.FreeImage_GetBPP(dib)
    if bpp not in [1,4,8]:
        raise Error('Unsupported bit depth for palette')
    if 1<<(bpp+2) != len(pal):
        raise Error('Invalid length of palette')
    ptr = _dll.FreeImage_GetPalette(dib)
    src = ctypes.create_string_buffer(pal)
    ctypes.memmove(ptr, src, 1<<(bpp+2))

mono8Palette = ''.join([chr(x)*4 for x in range(256)])

def SetPaletteMono8(dib):
    bpp = _dll.FreeImage_GetBPP(dib)
    if bpp != 8:
        raise Error('Mono8 palette requires bpp=8')
    ptr = _dll.FreeImage_GetPalette(dib)
    src = ctypes.create_string_buffer(mono8Palette)
    ctypes.memmove(ptr, src, 1<<(bpp+2))
    
# filetype functions

_dll.FreeImage_GetFIFFromFilename.restype = ctypes.c_int
_dll.FreeImage_GetFIFFromFilename.argtypes = (ctypes.c_char_p,)

def GetFIFFromFilename(filename):
    "Get the image format from the filename or extension"
    return _dll.FreeImage_GetFIFFromFilename(filename)

_dll.FreeImage_GetFileType.restype = ctypes.c_int
_dll.FreeImage_GetFileType.argtypes = (ctypes.c_char_p,
                                       ctypes.c_int)

def GetFileType(filename, size=0):
    """
    Return the file type for an image file or FIF_UNKNOWN if it
    cannot be determined
    """
    return _dll.FreeImage_GetFileType(filename, size)

# pixel access functions

_dll.FreeImage_GetBits.restype = ctypes.c_void_p
_dll.FreeImage_GetBits.argtypes = (FIBITMAP,)

def GetBits(dib):
    "Return a copy of the raw data"
    # determine size of image
    pitch = GetPitch(dib)
    height = GetHeight(dib)
    total_size = pitch * height
    # get a pointer to the data
    ptr = _dll.FreeImage_GetBits(dib)
    result = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_char*total_size)).contents
    # result = ctypes.string_at(ptr, total_size)
    return result

_dll.FreeImage_GetScanLine.restype = ctypes.c_void_p
_dll.FreeImage_GetScanLine.argtype = (FIBITMAP,
                                      ctypes.c_int)

def SetBits(dib, s):
    "Copy the image data from s into the image"
    # determine size of image
    pitch = GetPitch(dib)
    height = GetHeight(dib)
    total_size = pitch * height

    if len(s) != total_size:
        raise Error("Erroneous size for image data, expecting %d but got %d instead" % (total_size, len(s)))

    # get a pointer to the data
    ptr = _dll.FreeImage_GetBits(dib)

    # copy the data into a string buffer
    buf = ctypes.create_string_buffer(s, total_size)
    
    # copy that data into the dib
    ctypes.memmove(ptr, buf, total_size)

_dll.FreeImage_ConvertToRawBits.restype = None
_dll.FreeImage_ConvertToRawBits.argtypes = (ctypes.c_char_p,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_uint,
                                            ctypes.c_uint,
                                            ctypes.c_uint,
                                            ctypes.c_uint,
                                            ctypes.c_int)

def ConvertToRawBits(dib, pitch, bpp, red_mask, green_mask, blue_mask, topdown=False):
    # allocate a buffer for the output image
    if not dib:
        raise Error("Unallocated bitmap")
    line_size = (GetWidth(dib)*GetBPP(dib)+7)/8
    if pitch < 0:
        # auto-pitch
        pitch = line_size
    if line_size > pitch:
        raise Error("Pitch too small for image")
    mem_size = pitch * GetHeight(dib)
    if not mem_size:
        raise Error("Image has no size")
    bits = ctypes.create_string_buffer(mem_size)
    _dll.FreeImage_ConvertToRawBits(bits, dib, pitch, bpp,
                                    red_mask, green_mask, blue_mask,
                                    topdown)
    # retrieve the string
    return bits.raw

def ConvertFromRawBitsT(bits, fit, width, height, pitch, bpp=8,
                        red_mask=0, green_mask=0, blue_mask=0, topdown=False):
    dib = _dll.FreeImage_AllocateT(fit, width, height, bpp, 
                                   red_mask, green_mask, blue_mask)

    if not dib:
        raise Error("Unable to allocate new image for conversion from raw bits")
    if topdown:
        rows = range(height)
    else:
        rows = range(height-1, -1, -1)

    # obtain the number of bytes in a line
    linesize = _dll.FreeImage_GetLine(dib)

    if len(bits) < linesize*height:
        raise Error("Input data too small, need to be %d byte(s) long but got a string of size %d byte(s) instead" % (linesize*height, len(bits)))

    # make a copy of the data so that we can step through it
    data = ctypes.create_string_buffer(bits, len(bits))
    src = ctypes.addressof(data)
    for row in rows:
        dst = _dll.FreeImage_GetScanLine(dib, row)
        ctypes.memmove(dst, src, linesize)
        # step to the next line in the source
        src += pitch

    return dib
