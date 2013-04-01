import sys
from qcam import QCam
import Image

if len(sys.argv) != 4:
    print >> sys.stderr, "Usage: %s <exposure in microseconds> <output file name format> <number of frames>" % sys.argv[0]
    sys.exit(-1)

exposure_ms = int(sys.argv[1])
output_file_format = sys.argv[2]
num_frames = int(sys.argv[3])

# set up the camera
import QCam
QCam.LoadDriver()

# open the camera
cam = QCam.OpenCamera(QCam.ListCameras()[0])

# set the exposure
print >> sys.stderr, "Setting exposure to %d microsecond(s)" % exposure_ms
cam.settings.exposure = exposure_ms
cam.settings.imageFormat = 'mono16'
cam.settings.Flush()

cam.StartStreaming()

import time
start = time.time()
for frame_no in xrange(num_frames):
    # get a frame
    frame = cam.GrabFrame()
    print >> sys.stderr, "Creating %dx%d image" % (frame.width,frame.height)
    # create an image
    img = Image.fromstring('I;16',(frame.width,frame.height),frame.stringBuffer)
    try:
         output_file_name = output_file_format % (frame_no)
    except TypeError:
         output_file_name = output_file_format
    print >> sys.stderr, "Writing to "+output_file_name
    img.save(output_file_name)
    print >> sys.stderr, 'Finished writing'
end = time.time()
print '%f fps' % (num_frames / (end-start))

cam.StopStreaming()
cam.CloseCamera()
QCam.ReleaseDriver()
