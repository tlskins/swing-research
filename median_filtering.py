import numpy as np
import cv2
from IPython.display import Video
from collections import OrderedDict
from scipy.spatial import distance as dist
# %matplotlib inline
from matplotlib import pyplot as plt
from centroid_tracker import CentroidTracker

np.random.seed(42)

INPUT_VIDEO = './inputs/rallying_2_16_clip_0.mp4'
# INPUT_VIDEO = 'clip_0.mp4'
OUTPUT_VIDEO = './out/rallying_2_16_clip_0_yellow.mp4'
MEDIAN_FRAMES = 30
# TEST_BALL = 'ball_test.png'


def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def intersects(box1, box2):
    x1min = box1[0]
    x1max = box1[2]
    y1min = box1[1]
    y1max = box1[3]

    x2min = box2[0]
    x2max = box2[2]
    y2min = box2[1]
    y2max = box2[3]
    return (x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max)


def intersectsAny(box, boxes):
    for othBox in boxes:
        if intersects(box, othBox):
            return True
    return False


# load comparison ball
# ball = cv2.imread(TEST_BALL, cv2.COLOR_BGR2GRAY)
# ball = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)

video_stream = cv2.VideoCapture(INPUT_VIDEO)
frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * \
    np.random.uniform(size=MEDIAN_FRAMES)

# Store selected frames in an array
frames = []
for fid in frameIds:
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = video_stream.read()
    frames.append(frame)

video_stream.release()

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
# plt.imshow(fixColor(medianFrame))
# plt.pause(1)

avgFrame = np.average(frames, axis=0).astype(dtype=np.uint8)
# plt.imshow(fixColor(avgFrame))
# plt.pause(1)

sample_frame = frames[0]
# plt.imshow(fixColor(sample_frame))
# plt.pause(1)

grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
# plt.imshow(fixColor(grayMedianFrame))
# plt.pause(1)

graySample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
# plt.imshow(fixColor(graySample))
# plt.pause(1)

dframe = cv2.absdiff(graySample, grayMedianFrame)
# plt.imshow(fixColor(dframe))
# plt.pause(1)

blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
# plt.imshow(fixColor(blurred))
# plt.pause(1)

ret, tframe = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.imshow(fixColor(tframe))
# plt.pause(1)

(cnts, _) = cv2.findContours(tframe.copy(),
                             cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)

# for cnt in cnts:
#     x, y, w, h = cv2.boundingRect(cnt)
#     if y > 200:  # Disregard item that are the top of the picture
#         cv2.rectangle(sample_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# plt.imshow(fixColor(sample_frame))
# plt.pause(1)

# Create a new video stream and get total frame count
video_stream = cv2.VideoCapture(INPUT_VIDEO)
total_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
frame_w = round(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = round(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*"MP4V")
writer = cv2.VideoWriter(OUTPUT_VIDEO, codec, 30, (frame_w, frame_h))
ct = CentroidTracker()

frameCnt = 0
histories = []
while(frameCnt < total_frames-1):

    frameCnt += 1
    ret, frame = video_stream.read()

    print('frame count {}'.format(frameCnt))

    # Bound moving objects
    # https://github.com/cloudxlab/opencv-intro/blob/master/detect_moving_objects_video.ipynb
    # Convert current frame to grayscale
    # gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # Calculate absolute difference of current frame and
    # # the median frame
    # dframe = cv2.absdiff(gframe, grayMedianFrame)
    # # Gaussian
    # blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
    # # Thresholding to binarise
    # ret, tframe = cv2.threshold(blurred, 0, 255,
    #                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # # Identifying contours from the threshold
    # (cnts, _) = cv2.findContours(tframe.copy(),
    #                              cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)
    # # For each contour draw the bounding bos
    # moveRects = []
    # for idx, cnt in enumerate(cnts):
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     # if y > 200:  # Disregard items in the top of the picture
    #     #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     if w*h < 200 and w > 0.9*h and w < 1.1*h:
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #         label = 'area {} | h {} | w {}'.format(w*h, h, w)
    #         cv2.putText(frame, label, (x, y - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         moveRects.append([x, y, x+w, y+h])

    # Bound yellow contours
    # https://stackoverflow.com/questions/57262974/tracking-yellow-color-object-with-opencv-python
    hframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower = np.array([22, 93, 0], dtype="uint8")  # too low
    # lower = np.array([22, 150, 150], dtype="uint8") # too high
    # lower = np.array([22, 100, 100], dtype="uint8")  # pretty good
    lower = np.array([22, 93, 50], dtype="uint8")  # some noise
    upper = np.array([45, 255, 255], dtype="uint8")
    mask = cv2.inRange(hframe, lower, upper)

    yellowCnts = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellowCnts = yellowCnts[0] if len(yellowCnts) == 2 else yellowCnts[1]

    tracked = []
    yellow_dist = 0
    for cnt in yellowCnts:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        if area < 300 and area > 10 and w > 0.8*h and w < 1.2*h:
            box = [x, y, x+w, y+h]
            tracked.append(box)
            moved = len(histories) != 0
            for hist in histories:
                for obj in hist:
                    if intersects(obj, box):
                        moved = False
                        break

            if moved:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.putText(frame, "area {}".format(area), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if len(histories) == 3:
        histories.pop(0)
    histories.append(tracked)

    # Upload bounding boxes to centroid tracker
    # objects = ct.update(histories)
    # # loop over the tracked objects
    # for (objectID, centroid) in objects.items():
    #     # draw both the ID of the object and the centroid of the
    #     # object on the output frame
    #     text = "ID {}".format(objectID)
    #     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    writer.write(frame)

# Release video object
video_stream.release()
writer.release()
