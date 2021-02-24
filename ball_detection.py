import numpy as np
import cv2
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


INPUT_VIDEO = 'tim_ground_profile_wide_540p.mp4'
# INPUT_VIDEO = 'clip_0.mp4'
OUTPUT_VIDEO = 'ball_detection.mp4'
MEDIAN_FRAMES = 30
# TEST_BALL = 'ball_test.png'
BUFFER = 64

# Create a new video stream and get total frame count
video_stream = cv2.VideoCapture(INPUT_VIDEO)
total_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
frame_w = round(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = round(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*"MP4V")
writer = cv2.VideoWriter(OUTPUT_VIDEO, codec, 30, (frame_w, frame_h))

greenLower = (22, 93, 50)
greenUpper = (45, 255, 255)
# lower = np.array([22, 93, 50], dtype="uint8")  # some noise
# upper = np.array([45, 255, 255], dtype="uint8")
pts = deque(maxlen=BUFFER)

frameCnt = 0
while(frameCnt < total_frames-1):

    frameCnt += 1
    ret, frame = video_stream.read()

    print('frame count {}'.format(frameCnt))

    # resize the frame, blur it, and convert it to the HSV
    # color space
    # frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(BUFFER / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

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
    # hframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # lower = np.array([22, 93, 0], dtype="uint8")  # too low
    # # lower = np.array([22, 150, 150], dtype="uint8") # too high
    # # lower = np.array([22, 100, 100], dtype="uint8")  # pretty good
    # lower = np.array([22, 93, 50], dtype="uint8")  # some noise
    # upper = np.array([45, 255, 255], dtype="uint8")
    # mask = cv2.inRange(hframe, lower, upper)

    # yellowCnts = cv2.findContours(
    #     mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # yellowCnts = yellowCnts[0] if len(yellowCnts) == 2 else yellowCnts[1]

    # tracked = []
    # for cnt in yellowCnts:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     area = w*h
    #     if area < 300 and area > 10 and w > 0.8*h and w < 1.2*h:
    #         box = [x, y, x+w, y+h]
    #         tracked.append(box)
    #         moved = len(histories) != 0
    #         for hist in histories:
    #             for obj in hist:
    #                 if intersects(obj, box):
    #                     moved = False
    #                     break

    #         if moved:
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
    #             cv2.putText(frame, "area {}".format(area), (x, y - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # if len(histories) == 3:
    #     histories.pop(0)
    # histories.append(tracked)

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
