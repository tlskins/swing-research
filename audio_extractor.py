# Python code to convert video to audio
import moviepy.editor as mp

INPUT_VIDEO = './inputs/rallying_2_16.MOV'
OUTPUT_AUDIO = './out/rallying_2_16_clip_0.mp3'

video = mp.VideoFileClip(INPUT_VIDEO)
video_clip = video.subclip(0, 60)
video_clip.audio.write_audiofile(OUTPUT_AUDIO)
