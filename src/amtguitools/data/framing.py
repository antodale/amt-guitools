import numpy as np

from amtguitools.data.schema import STANDARD_TUNING

TABCNN_SAMPLE_RATE = 22050
TABCNN_HOP_LENGTH = 512
TABCNN_N_FRETS = 19


def note_frames(note, T, s_rate, hop_length):
    """
    First and last frame (both included) of a note, or None if it starts after the audio ends.
    """
    frames_per_second = s_rate / hop_length
    first = int(np.floor(note.onset * frames_per_second))   # frame that contains the onset
    last = int(np.floor(note.offset * frames_per_second))   # frame that contains the offset
    if first >= T:
        return None
    return first, min(last, T - 1)


def notes_to_tabs(notes, T, s_rate=TABCNN_SAMPLE_RATE, hop_length=TABCNN_HOP_LENGTH, n_frets=TABCNN_N_FRETS):
    """
    Build a (6, T) array with the fret played on each string in each frame (-1 = silence).

    notes : canonical notes table (uses the onset, offset, string and fret columns)
    T     : number of frames
    """
    # We start with an all silent array
    tablature = np.full((6, T), -1, dtype=np.int64)

    for note in notes.itertuples():
        if note.fret > n_frets:
            continue  # TabCNN has no class for this fret
        bounds = note_frames(note, T, s_rate, hop_length)
        if bounds is None:
            continue
        first, last = bounds

        # Write the fret; where notes overlap on a string, keep the lowest fret (like amt-tools)
        frames = tablature[note.string, first:last + 1]
        frames[(frames == -1) | (frames > note.fret)] = note.fret

    return tablature


def notes_to_multipitch(notes, T, s_rate=TABCNN_SAMPLE_RATE, hop_length=TABCNN_HOP_LENGTH,
                        tuning=STANDARD_TUNING, n_frets=TABCNN_N_FRETS):
    """
    Build a (44, T) array with 1 where a pitch sounds in a frame and 0 elsewhere.

    Row 0 is the open low E (pitch 40), the last row is high E at the last fret (pitch 83).
    Built from the notes, not the tablature: where two notes overlap on a string,
    the tablature keeps only one of them, but both pitches sound.
    """
    lowest = min(tuning)
    n_pitches = max(tuning) + n_frets - lowest + 1
    multi_pitch = np.zeros((n_pitches, T))

    for note in notes.itertuples():
        row = int(round(note.pitch)) - lowest
        if row < 0 or row >= n_pitches:
            continue  # pitch outside the guitar's range
        bounds = note_frames(note, T, s_rate, hop_length)
        if bounds is None:
            continue
        first, last = bounds
        multi_pitch[row, first:last + 1] = 1

    return multi_pitch
