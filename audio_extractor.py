# Python code to convert video to audio
import moviepy.editor as mp

INPUT_VIDEO = 'rallying_2_16.MOV'
OUTPUT_AUDIO = 'rallying_2_16.mp3'

# Insert Local Video File Path
clip = mp.VideoFileClip(INPUT_VIDEO)

# Insert Local Audio File Path
clip.audio.write_audiofile(OUTPUT_AUDIO)
