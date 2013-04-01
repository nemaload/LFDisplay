import sys
from qcam import QCam, QCamUtil
import Image

import struct
import numpy

if len(sys.argv) != 4:
    print >> sys.stderr, "Usage: %s <exposure in microseconds> <output file name format> <number of frames>" % sys.argv[0]
    sys.exit(-1)

exposure_ms = int(sys.argv[1])
output_file_format = sys.argv[2]
num_frames = int(sys.argv[3])

# set up the camera
QCam.ReleaseDriver()
QCam.LoadDriver()

# open the camera
cam = QCam.OpenCamera(QCam.ListCameras()[0])

# set the exposure
print >> sys.stderr, "Setting exposure to %d microsecond(s)" % exposure_ms
cam.settings.exposure = exposure_ms
cam.settings.imageFormat = 'mono16'
cam.settings.Flush()

# unused code
# create a camera queue
queue = QCamUtil.CameraQueue(cam)

# start capturing
queue.start(5)

# images = []

for frame_no in xrange(num_frames):
    print >> sys.stderr, "Waiting for frame %d/%d" % (frame_no+1,num_frames)
    # get the frame
    try:
        frame = queue.get(True, 1)
    except queue.Empty:
        continue

    try:
         output_file_name = output_file_format % (frame_no)
    except TypeError:
         output_file_name = output_file_format
    
    print >> sys.stderr, 'Creating %dx%d image' % (frame.width,frame.height)


    # create an image
    if output_file_name.endswith('tmp') or output_file_name.endswith('TMP'):
        # imagestack tmp file
        arr=numpy.fromstring(frame.stringBuffer, dtype='H', count=frame.width*frame.height)
        arr=arr.astype('f')
        print >> sys.stderr, 'Writing to '+output_file_name
        f=open(output_file_name,'wb')
        f.write(struct.pack('iiii',1,frame.width,frame.height,1))
        f.write(arr.tostring())
        f.close()
    else:
        # normal image file
        img = Image.fromstring('I;16',(frame.width,frame.height),frame.stringBuffer)
        print >> sys.stderr, 'Writing to '+output_file_name
        img.save(output_file_name)

    # put the frame back into the queue
    queue.put(frame)

# stop the queue when done
queue.stop()
