import numpy as np
import cv2
from IPython.display import Video
from collections import OrderedDict
from scipy.spatial import distance as dist
from contact_sound_detector import detect_contacts

np.random.seed(42)

INPUT_VIDEO = './inputs/rallying_2_16_clip_0.mp4'
OUTPUT_VIDEO = './out/rallying_2_16_clip_0_person.mp4'
MEDIAN_FRAMES = 30


def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# get median frame
video_stream = cv2.VideoCapture(INPUT_VIDEO)
frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * \
    np.random.uniform(size=MEDIAN_FRAMES)
frames = []
for fid in frameIds:
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = video_stream.read()
    frames.append(frame)
video_stream.release()
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# get median frame contours
avgFrame = np.average(frames, axis=0).astype(dtype=np.uint8)
sample_frame = frames[0]
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
graySample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
dframe = cv2.absdiff(graySample, grayMedianFrame)
blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
ret, tframe = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

(cnts, _) = cv2.findContours(tframe.copy(),
                             cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)

# for cnt in cnts:
#     x, y, w, h = cv2.boundingRect(cnt)
#     if y > 200:  # Disregard item that are the top of the picture
#         cv2.rectangle(sample_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Create a new video stream and get total frame count
video_stream = cv2.VideoCapture(INPUT_VIDEO)
total_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
frame_w = round(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = round(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('total frames {} width {} height {}'.format(
    total_frames, frame_w, frame_h))
codec = cv2.VideoWriter_fourcc(*"MP4V")
writer = cv2.VideoWriter(OUTPUT_VIDEO, codec, 30, (frame_w, frame_h))

frameCnt = 0
histories = []
while(frameCnt < total_frames-1):

    frameCnt += 1
    ret, frame = video_stream.read()

    print('frame count {}'.format(frameCnt))

    # Bound moving objects
    # https://github.com/cloudxlab/opencv-intro/blob/master/detect_moving_objects_video.ipynb
    # Convert current frame to grayscale
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and
    # the median frame
    dframe = cv2.absdiff(gframe, grayMedianFrame)
    # Gaussian
    blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
    # Thresholding to binarise
    ret, tframe = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Identifying contours from the threshold
    (cnts, _) = cv2.findContours(tframe.copy(),
                                 cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)
    # For each contour draw the bounding bos
    maxCnt = [0, 0, 0, 0]
    for idx, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > maxCnt[2]*maxCnt[3]:
            maxCnt = [x, y, w, h]

    x, y, w, h = maxCnt
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = 'ratio {}'.format(round(h/w, 2))
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    writer.write(frame)

# Release video object
writer.release()
video_stream.release()
