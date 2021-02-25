import numpy as np
import cv2
from IPython.display import Video
from collections import OrderedDict
from scipy.spatial import distance as dist
from contact_sound_detector import detect_contacts

np.random.seed(42)

INPUT_VIDEO = './inputs/rallying_2_16_clip_0.mp4'
INPUT_AUDIO = './inputs/rallying_2_16_clip_0.mp3'
MEDIAN_FRAMES = 30
FPS = 30
PRE_STRIKE_FRAMES = 25
POST_STRIKE_FRAMES = 35


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

contacts = detect_contacts(INPUT_AUDIO)
print(contacts)
swing_num = 0
for timestamp in contacts:
    # calculate swing start and end frames
    contact_frame = round(timestamp * FPS)
    st_frame = contact_frame - PRE_STRIKE_FRAMES + 1
    if st_frame < 1:
        st_frame = 1
    end_frame = contact_frame + POST_STRIKE_FRAMES
    if end_frame > total_frames:
        end_frame = round(total_frames)

    # write swing frames
    out_filename = './out/swing_{}.mp4'.format(swing_num)
    writer = cv2.VideoWriter(out_filename, codec, FPS, (frame_w, frame_h))
    for frame_num in range(st_frame, end_frame):
        video_stream.set(1, frame_num)
        ret, frame = video_stream.read()

        # Bound moving objects
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dframe = cv2.absdiff(gframe, grayMedianFrame)
        blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
        ret, tframe = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        (cnts, _) = cv2.findContours(tframe.copy(),
                                     cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)
        max_cnt = [0, 0, 0, 0]
        for idx, cnt in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h > max_cnt[2]*max_cnt[3]:
                max_cnt = [x, y, w, h]

        # bound largest contour
        x, y, w, h = max_cnt
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = 'area {}'.format(w*h)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        max_cnt = [0, 0, 0, 0]
        writer.write(frame)

    writer.release()
    swing_num += 1

# Release video object
video_stream.release()
