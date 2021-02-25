from moviepy import editor


INPUT_FILE = './inputs/rallying_2_16.MOV'
OUTPUT_FILE = './out/rallying_2_16_clip_0.mp4'

video = editor.VideoFileClip(INPUT_FILE)
duration = video.duration

video_clip = video.subclip(0, 60)
video_clip.write_videofile(OUTPUT_FILE)
