from moviepy import editor

# rallying_ph_2_16

INPUT_FILE = './inputs/rallying_ph_2_16.mov'

video = editor.VideoFileClip(INPUT_FILE)
# video_clip = video.subclip(0, 60)
# video_clip.write_videofile('./inputs/rallying_ph_2_16_clip_0.mp4', audio=False)
# video_clip = video.subclip(60, 120)
# video_clip.write_videofile('./inputs/rallying_ph_2_16_clip_1.mp4', audio=False)
video_clip = video.subclip(0, 300)
video_clip = video_clip.fx(editor.vfx.speedx, 1.5)

video_clip.write_videofile('./inputs/rallying_ph_2_16.mp4', audio=False)
