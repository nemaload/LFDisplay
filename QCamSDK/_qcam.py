import ctypes
import platform

class Error(Exception):
    pass

if platform.system() in ('Windows','Microsoft'):
    _dll = ctypes.WinDLL('QCamDriver.dll')
    FUNCTYPE = ctypes.WINFUNCTYPE
elif platform.system() == 'Darwin':
    _dll = ctypes.CDLL('QCam.framework/Versions/Current/QCam')
    FUNCTYPE = ctypes.CFUNCTYPE
else:
    raise Error('The QCam Python wrapper is not supported on '+platform.system())

# camera description
class QCam_CamListItem(ctypes.Structure):
    _fields_ = [('cameraId',ctypes.c_ulong),
                ('cameraType',ctypes.c_ulong),
                ('uniqueId',ctypes.c_ulong),
                ('isOpen',ctypes.c_ulong),
                ('_reserved',ctypes.c_ulong*10)]
# frame description
class QCam_Frame(ctypes.Structure):
    _fields_ = [('pBuffer',ctypes.c_void_p),
                ('bufferSize',ctypes.c_ulong),
                ('format',ctypes.c_ulong),
                ('width',ctypes.c_ulong),
                ('height',ctypes.c_ulong),
                ('size',ctypes.c_ulong),
                ('bits',ctypes.c_ushort),
                ('frameNumber',ctypes.c_ushort),
                ('bayerPattern',ctypes.c_ulong),
                ('errorCode',ctypes.c_ulong),
                ('timeStamp',ctypes.c_ulong),
                ('_reserved',ctypes.c_ulong*8)]
# settings
class QCam_Settings(ctypes.Structure):
    _fields_ = [('size',ctypes.c_ulong),
                ('_private_data',ctypes.c_ulong * 64)]
    def __init__(self, *args, **kwds):
        ctypes.Structure.__init__(self, *args, **kwds)
        self.size = ctypes.sizeof(ctypes.c_ulong)*65

# types
QCam_Err=ctypes.c_int
QCam_Info=ctypes.c_int
QCam_Handle=ctypes.c_void_p
UNSIGNED64=ctypes.c_ulonglong

QCam_Param=ctypes.c_int
QCam_ParamS32=ctypes.c_int
QCam_Param64=ctypes.c_int

# callback function
QCam_AsyncCallback = FUNCTYPE(None,
                              ctypes.c_void_p, ctypes.c_ulong, QCam_Err, ctypes.c_ulong)

# function prototypes

_dll.QCam_LoadDriver.restype = QCam_Err
_dll.QCam_LoadDriver.argtypes = []

_dll.QCam_ReleaseDriver.restype = None
_dll.QCam_ReleaseDriver.argtypes = []

_dll.QCam_LibVersion.restype = QCam_Err
_dll.QCam_LibVersion.argtypes = [ctypes.POINTER(ctypes.c_ushort),
                                 ctypes.POINTER(ctypes.c_ushort),
                                 ctypes.POINTER(ctypes.c_ushort)]

_dll.QCam_ListCameras.restype = QCam_Err
_dll.QCam_ListCameras.argtypes = [ctypes.POINTER(QCam_CamListItem),
                                  ctypes.POINTER(ctypes.c_ulong)]

_dll.QCam_OpenCamera.restype = QCam_Err
_dll.QCam_OpenCamera.argtypes = [ctypes.c_ulong,
                                 ctypes.POINTER(QCam_Handle)]

_dll.QCam_CloseCamera.restype = QCam_Err
_dll.QCam_CloseCamera.argtypes = [QCam_Handle]

_dll.QCam_GetSerialString.restype = QCam_Err
_dll.QCam_GetSerialString.argtypes = [QCam_Handle,
                                      ctypes.c_char_p,
                                      ctypes.c_ulong]

_dll.QCam_GetInfo.restype = QCam_Err
_dll.QCam_GetInfo.argtypes = [QCam_Handle,
                              QCam_Info,
                              ctypes.POINTER(ctypes.c_ulong)]

_dll.QCam_ReadDefaultSettings.restype = QCam_Err
_dll.QCam_ReadDefaultSettings.argtypes = [QCam_Handle,
                                          ctypes.POINTER(QCam_Settings)]

_dll.QCam_ReadSettingsFromCam.restype = QCam_Err
_dll.QCam_ReadSettingsFromCam.argtypes = [QCam_Handle,
                                          ctypes.POINTER(QCam_Settings)]

_dll.QCam_SendSettingsToCam.restype = QCam_Err
_dll.QCam_SendSettingsToCam.argtypes = [QCam_Handle,
                                        ctypes.POINTER(QCam_Settings)]

_dll.QCam_PreflightSettings.restype = QCam_Err
_dll.QCam_PreflightSettings.argtypes = [QCam_Handle,
                                        ctypes.POINTER(QCam_Settings)]

_dll.QCam_TranslateSettings.restype = QCam_Err
_dll.QCam_TranslateSettings.argtypes = [QCam_Handle,
                                        ctypes.POINTER(QCam_Settings)]

_dll.QCam_GetParam.restype = QCam_Err
_dll.QCam_GetParam.argtypes = [ctypes.POINTER(QCam_Settings),
                               QCam_Param,
                               ctypes.POINTER(ctypes.c_ulong)]

_dll.QCam_GetParamS32.restype = QCam_Err
_dll.QCam_GetParamS32.argtypes = [ctypes.POINTER(QCam_Settings),
                                  QCam_ParamS32,
                                  ctypes.POINTER(ctypes.c_long)]

_dll.QCam_GetParam64.restype = QCam_Err
_dll.QCam_GetParam64.argtypes = [ctypes.POINTER(QCam_Settings),
                                 QCam_Param64,
                                 ctypes.POINTER(UNSIGNED64)]

_dll.QCam_SetParam.restype = QCam_Err
_dll.QCam_SetParam.argtypes = [ctypes.POINTER(QCam_Settings),
                               QCam_Param,
                               ctypes.c_ulong]

_dll.QCam_SetParamS32.restype = QCam_Err
_dll.QCam_SetParamS32.argtypes = [ctypes.POINTER(QCam_Settings),
                                  QCam_ParamS32,
                                  ctypes.c_long]

_dll.QCam_SetParam64.restype = QCam_Err
_dll.QCam_SetParam64.argtypes = [ctypes.POINTER(QCam_Settings),
                                 QCam_Param64,
                                 ctypes.c_ulonglong]

_dll.QCam_GetParamMin.restype = QCam_Err
_dll.QCam_GetParamMin.argtypes = [ctypes.POINTER(QCam_Settings),
                                  QCam_Param,
                                  ctypes.POINTER(ctypes.c_ulong)]

_dll.QCam_GetParamS32Min.restype = QCam_Err
_dll.QCam_GetParamS32Min.argtypes = [ctypes.POINTER(QCam_Settings),
                                     QCam_ParamS32,
                                     ctypes.POINTER(ctypes.c_long)]

_dll.QCam_GetParam64Min.restype = QCam_Err
_dll.QCam_GetParam64Min.argtypes = [ctypes.POINTER(QCam_Settings),
                                    QCam_Param64,
                                    ctypes.POINTER(UNSIGNED64)]

_dll.QCam_GetParamMax.restype = QCam_Err
_dll.QCam_GetParamMax.argtypes = [ctypes.POINTER(QCam_Settings),
                                  QCam_Param,
                                  ctypes.POINTER(ctypes.c_ulong)]

_dll.QCam_GetParamS32Max.restype = QCam_Err
_dll.QCam_GetParamS32Max.argtypes = [ctypes.POINTER(QCam_Settings),
                                     QCam_ParamS32,
                                     ctypes.POINTER(ctypes.c_long)]

_dll.QCam_GetParam64Max.restype = QCam_Err
_dll.QCam_GetParam64Max.argtypes = [ctypes.POINTER(QCam_Settings),
                                    QCam_Param64,
                                    ctypes.POINTER(UNSIGNED64)]

_dll.QCam_GetParamSparseTable.restype = QCam_Err
_dll.QCam_GetParamSparseTable.argtypes = [ctypes.POINTER(QCam_Settings),
                                          QCam_Param,
                                          ctypes.POINTER(ctypes.c_ulong),
                                          ctypes.POINTER(ctypes.c_int)]

_dll.QCam_GetParamSparseTableS32.restype = QCam_Err
_dll.QCam_GetParamSparseTableS32.argtypes = [ctypes.POINTER(QCam_Settings),
                                             QCam_ParamS32,
                                             ctypes.POINTER(ctypes.c_long),
                                             ctypes.POINTER(ctypes.c_int)]

_dll.QCam_GetParamSparseTable64.restype = QCam_Err
_dll.QCam_GetParamSparseTable64.argtypes = [ctypes.POINTER(QCam_Settings),
                                            QCam_Param64,
                                            ctypes.POINTER(UNSIGNED64),
                                            ctypes.POINTER(ctypes.c_int)]

_dll.QCam_IsSparseTable.restype = QCam_Err
_dll.QCam_IsSparseTable.argtypes = [ctypes.POINTER(QCam_Settings),
                                    QCam_Param]

_dll.QCam_IsSparseTableS32.restype = QCam_Err
_dll.QCam_IsSparseTableS32.argtypes = [ctypes.POINTER(QCam_Settings),
                                       QCam_ParamS32]

_dll.QCam_IsSparseTable64.restype = QCam_Err
_dll.QCam_IsSparseTable64.argtypes = [ctypes.POINTER(QCam_Settings),
                                      QCam_Param64]

_dll.QCam_IsRangeTable.restype = QCam_Err
_dll.QCam_IsRangeTable.argtypes = [ctypes.POINTER(QCam_Settings),
                                    QCam_Param]

_dll.QCam_IsRangeTableS32.restype = QCam_Err
_dll.QCam_IsRangeTableS32.argtypes = [ctypes.POINTER(QCam_Settings),
                                       QCam_ParamS32]

_dll.QCam_IsRangeTable64.restype = QCam_Err
_dll.QCam_IsRangeTable64.argtypes = [ctypes.POINTER(QCam_Settings),
                                      QCam_Param64]

_dll.QCam_IsParamSupported.restype = QCam_Err
_dll.QCam_IsParamSupported.argtypes = [QCam_Handle,
                                       QCam_Param]

_dll.QCam_IsParamS32Supported.restype = QCam_Err
_dll.QCam_IsParamS32Supported.argtypes = [QCam_Handle,
                                          QCam_ParamS32]

_dll.QCam_IsParam64Supported.restype = QCam_Err
_dll.QCam_IsParam64Supported.argtypes = [QCam_Handle,
                                         QCam_Param64]

_dll.QCam_SetStreaming.restype = QCam_Err
_dll.QCam_SetStreaming.argtypes = [QCam_Handle,
                                   ctypes.c_ulong]

_dll.QCam_Trigger.restype = QCam_Err
_dll.QCam_Trigger.argtypes = [QCam_Handle]

_dll.QCam_Abort.restype = QCam_Err
_dll.QCam_Abort.argtypes = [QCam_Handle]

_dll.QCam_GrabFrame.restype = QCam_Err
_dll.QCam_GrabFrame.argtypes = [QCam_Handle,
                                ctypes.POINTER(QCam_Frame)]
                                
_dll.QCam_QueueFrame.restype = QCam_Err
_dll.QCam_QueueFrame.argtypes = [QCam_Handle,
                                 ctypes.POINTER(QCam_Frame),
                                 QCam_AsyncCallback,
                                 ctypes.c_ulong,
                                 ctypes.c_void_p,
                                 ctypes.c_ulong]

_dll.QCam_QueueSettings.restype = QCam_Err
_dll.QCam_QueueSettings.argtypes = [QCam_Handle,
                                    ctypes.POINTER(QCam_Settings),
                                    QCam_AsyncCallback,
                                    ctypes.c_ulong,
                                    ctypes.c_void_p,
                                    ctypes.c_ulong]
