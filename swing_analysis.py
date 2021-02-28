import numpy as np
import cv2
from contact_sound_detector import detect_contacts
from pitch_detector import detect_pitches

np.random.seed(42)

# rallying_ph_2_16_clip_0

INPUT_VIDEO = './inputs/rallying_ph_2_16_clip_1.mp4'
INPUT_AUDIO = './inputs/rallying_ph_2_16_clip_1.mp3'
OUTPUT_VIDEO = './out/rallying_ph_2_16_clip_1_analysis.mp4'
MEDIAN_FRAMES = 90
FPS = 30
PRE_STRIKE_FRAMES = 30
POST_STRIKE_FRAMES = 30
# CUT_WIDTH = 558
# CUT_HEIGHT = 314
CUT_WIDTH = 711
CUT_HEIGHT = 400

frame_w = None
frame_h = None


def clipFrameBox(bodyBox):
    centerX = bodyBox[0] + round((bodyBox[2]-bodyBox[0]) / 2)
    centerY = bodyBox[1] + round((bodyBox[3]-bodyBox[1]) / 2)
    minX = centerX - round(CUT_WIDTH/2)
    maxX = centerX + round(CUT_WIDTH/2)
    minY = centerY - round(CUT_HEIGHT/2)
    maxY = centerY + round(CUT_HEIGHT/2)

    if minX < 0:
        maxX += 0 - minX
        minX = 0
    elif maxX > frame_w:
        minX -= maxX - frame_w
        maxX = frame_w

    if minY < 0:
        maxY += 0 - minY
        minY = 0
    elif maxY > frame_h:
        minY -= maxY - frame_h
        maxY = frame_h

    return [minX, minY, maxX, maxY]


def maxContourBox(frame):
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dframe = cv2.absdiff(gframe, grayMedianFrame)
    blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
    _, tframe = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (cnts, _) = cv2.findContours(tframe.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = [0, 0, 0, 0]
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > max_cnt[2]*max_cnt[3]:
            max_cnt = [x, y, w, h]

    return [max_cnt[0], max_cnt[1], max_cnt[0]+max_cnt[2], max_cnt[1]+max_cnt[3]]


def boundContours(frame):
    # gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # dframe = cv2.absdiff(gframe, grayMedianFrame)
    # blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
    # _, tframe = cv2.threshold(blurred, 0, 255,
    #                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # (cnts, _) = cv2.findContours(tframe.copy(),
    #                              cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)

    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dframe = cv2.absdiff(gframe, hsvMedianFrame)
    _, _, dframe = cv2.split(dframe)  # to grayscale
    blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
    _, tframe = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (cnts, _) = cv2.findContours(tframe.copy(),
                                 cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)

    # get body contour
    maxCnt = None
    maxArea = 0
    maxX = 0
    maxY = 0
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > maxArea:
            maxArea = w*h
            maxCnt = cnt
            maxX = x
            maxY = y

    # get all contours close to body
    bodyCnts = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        point1 = np.array((1, x, y))
        point2 = np.array((1, maxX, maxY))
        dist = np.linalg.norm(point1 - point2)
        if dist <= 100:
            bodyCnts.append(cnt)
    cv2.drawContours(frame, bodyCnts, -1, (0, 255, 0), 2)

    # Find contours in thresh_gray after closing the gaps
    # tframe = cv2.morphologyEx(
    #     tframe, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    # _, contours = cv2.findContours(
    #     tframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # for c in contours:
    #     if len(contours) > 0:
    #         # area = cv2.contourArea(c)
    #         # # Small contours are ignored.
    #         # if area < 500:
    #         #     cv2.fillPoly(tframe, pts=[c], color=0)
    #         #     continue

    #         rect = cv2.minAreaRect(c)
    #         box = cv2.boxPoints(rect)
    #         # convert all coordinates floating point values to int
    #         box = np.int0(box)
    #         cv2.drawContours(frame, [box], 0, (0, 255, 0), 1)

    # calculate extent
    area = cv2.contourArea(maxCnt)
    x, y, w, h = cv2.boundingRect(maxCnt)
    rect_area = w*h
    extent = float(area)/rect_area
    cv2.putText(frame, 'extent {}'.format(extent), (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # calculate orientation
    (ox, oy), (MA, ma), angle = cv2.fitEllipse(maxCnt)
    text_color = (0, 0, 0)
    cv2.putText(frame, 'MA {}'.format(MA), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    cv2.putText(frame, 'ma {}'.format(ma), (50, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    cv2.putText(frame, 'angle {}'.format(angle), (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    body_color = (255, 255, 0) if MA >= 100 else (0, 255, 0)
    cv2.drawContours(frame, [maxCnt], -1, body_color, 2)


# read video
video_stream = cv2.VideoCapture(INPUT_VIDEO)
total_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
frame_w = round(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = round(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('total frames {} width {} height {}'.format(
    total_frames, frame_w, frame_h))
codec = cv2.VideoWriter_fourcc(*"MP4V")

# get median frame
frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * \
    np.random.uniform(size=MEDIAN_FRAMES)
frames = []
for fid in frameIds:
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = video_stream.read()
    frames.append(frame)
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# get median frame contours
avgFrame = np.average(frames, axis=0).astype(dtype=np.uint8)
sample_frame = frames[0]
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
hsvMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2HSV)
graySample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
dframe = cv2.absdiff(graySample, grayMedianFrame)
blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
ret, tframe = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(cnts, _) = cv2.findContours(tframe.copy(),
                             cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)

# get pitches
# pitch_times, pitches, volumes = detect_pitches(INPUT_AUDIO)
# print(pitch_times)
# pitch_idx = 0

writer = cv2.VideoWriter(OUTPUT_VIDEO, codec, 30, (frame_w, frame_h))
frame_num = 0
history = []

while frame_num < total_frames-1:
    frame_num += 1
    print('frame_num {}'.format(frame_num))

    (grabbed, frame) = video_stream.read()
    if not grabbed:
        continue

    # body_min_x, body_min_y, body_max_x, body_max_y = maxContourBox(frame)
    # w = body_max_x - body_min_x
    # h = body_max_y - body_min_y
    # ratio = round(w/h, 2)
    # delta = 0
    # if len(history) == 0:
    #     history = [w, w, w, w]
    # else:
    #     history.pop(0)
    #     history.append(w)
    #     last = history[0]
    #     for h in history:
    #         delta += abs(h-last)
    #         last = h

    boundContours(frame)
    # timestamp = round(frame_num / FPS, 2)
    # text_color = (255, 255, 0)
    # cv2.putText(frame, 'seconds {}'.format(timestamp), (50, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    # cv2.putText(frame, 'area {}'.format(w*h), (50, 65),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    # cv2.putText(frame, 'ratio {}'.format(ratio), (50, 80),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    # cv2.putText(frame, 'delta {}'.format(delta), (50, 95),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    # pitch = 0
    # if pitch_idx < len(pitch_times):
    #     if timestamp >= pitch_times[pitch_idx] - 0.2 and timestamp <= pitch_times[pitch_idx]:
    #         pitch = pitches[pitch_idx]
    #     elif timestamp > pitch_times[pitch_idx]:
    #         pitch_idx += 1
    # cv2.putText(frame, 'pitch {} vol {}'.format(pitch, volumes[pitch_idx]), (50, 110),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # if pitch >= 50 and delta > 20 and ratio >= 1.0:
    #     cv2.putText(frame, 'swing detected!', (50, 125),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    writer.write(frame)

writer.release()
video_stream.release()
