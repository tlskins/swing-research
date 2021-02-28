import sys
from aubio import source
from aubio import pitch as get_pitch
import numpy as num

MIN_SWING_BUFF = 1.5  # maybe 1.0 for wall and 1.25 for rallying
MIN_CONTACT_PITCH = 50
SAMPLE_PITCH_RATE = 0.25
SWING_WINDOW = 0.75


def detect_pitches(filename):
    print(filename)
    downsample = 1
    samplerate = 44100 // downsample
    win_s = 4096 // downsample  # fft size
    hop_s = 512 // downsample  # hop size

    s = source(filename, samplerate, hop_s)
    samplerate = s.samplerate

    tolerance = 0.8

    pitch_o = get_pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    total_frames = 0
    times = []
    pitches = []
    volumes = []
    confidences = []
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        pitch = int(round(pitch, 2))
        timestamp = total_frames / float(samplerate)
        confidence = pitch_o.get_confidence()
        # if confidence < 0.1:
        #     pitch = 0
        # print("%f %f" % (timestamp, pitch))
        total_frames += read
        if pitch >= MIN_CONTACT_PITCH:
            times.append(timestamp)
            pitches.append(pitch)
            volume = num.sum(samples**2)/len(samples)*100
            volume = "{:4f}".format(volume)
            volumes.append(volume)
            confidences.append(confidence)

        if read < hop_s:
            break

    return [times, pitches, volumes]


if __name__ == "__main__":
    pitches = detect_pitches(sys.argv[1])
    print(pitches)
