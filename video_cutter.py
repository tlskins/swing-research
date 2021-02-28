from moviepy import editor

# rallying_2_16

INPUT_FILE = './inputs/rallying_2_16.mov'

video = editor.VideoFileClip(INPUT_FILE)
# video_clip = video.subclip(0, 60)
# video_clip.write_videofile('./inputs/rallying_2_16_clip_0.mp4', audio=False)
# video_clip = video.subclip(60, 120)
# video_clip.write_videofile('./inputs/rallying_2_16_clip_1.mp4', audio=False)
video_clip = video.subclip(120, 180)
video_clip.write_videofile('./inputs/rallying_2_16_clip_2.mp4', audio=False)
