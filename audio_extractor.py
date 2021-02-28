# Python code to convert video to audio
import moviepy.editor as mp

INPUT_VIDEO = './inputs/rallying_2_16.mov'
OUTPUT_AUDIO = './out/rallying_2_16_clip_0.wav'

video = mp.VideoFileClip(INPUT_VIDEO)
video_clip = video.subclip(0, 60)
video_clip.audio.write_audiofile(OUTPUT_AUDIO, codec='pcm_s16le')
