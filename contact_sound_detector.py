import sys
from aubio import source
from aubio import pitch as get_pitch

MIN_SWING_BUFF = 1.0
MIN_CONTACT_PITCH = 80
SAMPLE_PITCH = 0.25


def detect_contacts(filename):
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
    max_pitch = 0
    last_pitch = 0
    last_contact = 0
    contacts = []
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        # pitch = int(round(pitch))
        # confidence = pitch_o.get_confidence()
        timestamp = total_frames / float(samplerate)
        # if confidence < 0.8: pitch = 0.
        # print("%f %f" % (timestamp, pitch))
        total_frames += read

        if pitch > max_pitch:
            max_pitch = pitch

        # get max pitch
        if timestamp - last_pitch >= SAMPLE_PITCH:
            # contact detected
            if max_pitch > MIN_CONTACT_PITCH and timestamp >= last_contact + MIN_SWING_BUFF:
                contacts.append(round(timestamp, 4))
                last_contact = timestamp
            last_pitch = timestamp
            max_pitch = 0

        if read < hop_s:
            break

    return contacts


if __name__ == "__main__":
    contacts = detect_contacts(sys.argv[1])
    print(contacts)
