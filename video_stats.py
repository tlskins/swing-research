import cv2

INPUT_FILE = './inputs/tim_ground_profile_wide_540p_clip_0_swing_1.mp4'

video_stream = video_stream = cv2.VideoCapture(INPUT_FILE)
total_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
frame_w = round(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = round(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('total frames {} width {} height {}'.format(
    total_frames, frame_w, frame_h))
