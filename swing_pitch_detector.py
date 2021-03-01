import numpy as np
import cv2
from contact_sound_detector import detect_contacts
from pitch_detector import detect_pitches

np.random.seed(42)

INPUT_VIDEO = './inputs/rallying_2_16_clip_1.mp4'
INPUT_AUDIO = './inputs/rallying_2_16_clip_1.mp3'
MEDIAN_FRAMES = 30
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
                                 cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)
    max_cnt = [0, 0, 0, 0]
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > max_cnt[2]*max_cnt[3]:
            max_cnt = [x, y, w, h]

    return [max_cnt[0], max_cnt[1], max_cnt[0]+max_cnt[2], max_cnt[1]+max_cnt[3]]


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
graySample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
dframe = cv2.absdiff(graySample, grayMedianFrame)
blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
ret, tframe = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

(cnts, _) = cv2.findContours(tframe.copy(),
                             cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)

# get contacts
contacts = detect_contacts(INPUT_AUDIO)
print(contacts)
swing_num = 0
for timestamp in contacts:
    # calculate swing start and end frames
    contact_frame_num = round(timestamp * FPS)
    st_frame = contact_frame_num - PRE_STRIKE_FRAMES + 1
    st_frame = 1 if st_frame < 1 else st_frame
    end_frame = contact_frame_num + POST_STRIKE_FRAMES
    end_frame = round(total_frames) if end_frame > total_frames else end_frame

    # get clip frame box
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, contact_frame_num)
    _, contact_frame = video_stream.read()
    contact_body_box = maxContourBox(contact_frame)
    minX, minY, maxX, maxY = clipFrameBox(contact_body_box)
    print('clip box {} {} {} {}'.format(minX, minY, maxX, maxY))

    # write swing frames
    out_filename = './out/swing_{}.mp4'.format(swing_num)
    writer = cv2.VideoWriter(out_filename, codec, FPS, (maxX-minX, maxY-minY))

    frame_num = st_frame
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    history = []
    while frame_num <= end_frame:
        ret, frame = video_stream.read()
        body_min_x, _, body_max_x, _ = maxContourBox(frame)
        w = body_max_x - body_min_x
        delta = 0
        if len(history) == 0:
            history = [w, w, w, w]
        else:
            history.pop(0)
            history.append(w)
            last = history[0]
            for h in history:
                delta += abs(h-last)
                last = h
        frame = frame[minY:maxY, minX:maxX, :]
        # cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 255, 0), 2)
        # cv2.putText(frame, 'area {}'.format(w*h), (x, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, 'seconds {} area {} ratio {} delta {}'.format(round(frame_num / FPS, 2), w*h, round(w / h, 2) delta), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        writer.write(frame)
        frame_num += 1

    writer.release()
    swing_num += 1

# Release video object
video_stream.release()
