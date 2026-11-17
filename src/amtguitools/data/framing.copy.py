"""
Framing: canonical notes -> frame-level targets for frame-level models such as TabCNN.

The canonical notes (see schema.py) are events in seconds. Frame-level models predict
one value per frame instead, so their ground truth must live on the same frame grid
as the audio features. This module builds that ground truth:

    tablature    (S x T)     fret played on each string in each frame, -1 = silent
    multi_pitch  (P x T)     1 if a pitch sounds (on any string) in a frame, else 0

    S = number of strings, T = number of frames,
    P = number of pitches the instrument can play (lowest open string .. highest string + num_frets)

The rules reproduce amt-tools (tools/utils.py: notes_to_multi_pitch, filter_notes,
stacked_multi_pitch_to_tablature, stacked_multi_pitch_to_multi_pitch) so that the
targets are identical to those used to train the TabCNN baseline:

    - Frame k starts at k * hop_length / sample_rate seconds.
    - The number of frames is 1 + num_samples // hop_length (the feature extractor's count).
    - A note is active from the last frame starting at or before its onset, up to and
      INCLUDING the last frame starting at or before its offset.
    - Notes are ignored if their rounded pitch is outside the instrument's range,
      if they start after the end of the grid, or if they end before it starts.
    - Tablature ignores frets above num_frets on each string. Those notes still count
      in multi_pitch if their pitch is within the instrument's range.
    - If two notes overlap on one string, the lowest fret wins.

Based on amt-tools by Frank Cwitkowitz (https://github.com/cwitkowitz/amt-tools).
See THIRD_PARTY_LICENSES.
"""

import math
from dataclasses import dataclass

import numpy as np

from .schema import STANDARD_TUNING, load_notes

# Settings used by the TabCNN baseline (amt-tools/examples/papers/tabcnn.py)
TABCNN_SAMPLE_RATE = 22050
TABCNN_HOP_LENGTH = 512
TABCNN_NUM_FRETS = 19


# ----------------------------------------------------------------------------
# Frame grid
# ----------------------------------------------------------------------------

def num_frames_for_samples(num_samples, hop_length=TABCNN_HOP_LENGTH):
    """Number of frames a feature extractor produces for audio of num_samples samples."""
    if num_samples <= 0:
        return 0
    return 1 + num_samples // hop_length


def num_samples_after_resampling(num_samples, original_rate, target_rate):
    """Length of audio after resampling, as computed by librosa (ceil of the length ratio)."""
    if original_rate == target_rate:
        return num_samples
    return int(math.ceil(num_samples * target_rate / original_rate))


def num_frames_for_audio_file(audio_path, sample_rate=TABCNN_SAMPLE_RATE, hop_length=TABCNN_HOP_LENGTH):
    """
    Number of frames for an audio file once loaded at sample_rate, without loading the audio.
    Only the file header is read.
    """
    import soundfile  # imported here so the rest of the module does not need it

    info = soundfile.info(str(audio_path))  # older soundfile versions reject Path objects
    num_samples = num_samples_after_resampling(info.frames, info.samplerate, sample_rate)
    return num_frames_for_samples(num_samples, hop_length)


def frame_times(num_frames, sample_rate=TABCNN_SAMPLE_RATE, hop_length=TABCNN_HOP_LENGTH):
    """Start time in seconds of each frame."""
    return np.arange(num_frames) * hop_length / sample_rate


def pitch_range(tuning=STANDARD_TUNING, num_frets=TABCNN_NUM_FRETS):
    """Lowest and highest MIDI pitch the instrument can play (inclusive)."""
    return int(min(tuning)), int(max(tuning)) + num_frets


# ----------------------------------------------------------------------------
# Notes -> frames
# ----------------------------------------------------------------------------

@dataclass
class FramingStats:
    """What happened to the notes of one track while framing."""

    num_notes: int = 0
    notes_used: int = 0              # notes that made it onto the frame grid
    out_of_range_pitch: int = 0      # rounded pitch outside the instrument's range: ignored
    outside_grid: int = 0            # start after the last frame / end before the first: ignored
    above_num_frets: int = 0         # in multi_pitch, but not in tablature
    overlapping_frames: int = 0      # (string, frame) pairs where several frets were active


def note_frame_bounds(onsets, offsets, num_frames, sample_rate=TABCNN_SAMPLE_RATE,
                      hop_length=TABCNN_HOP_LENGTH):
    """
    First and last active frame (both inclusive) of each note.

    first = last frame starting at or before the onset (clipped to frame 0)
    last  = last frame starting at or before the offset (clipped to the last frame)
    """
    times = frame_times(num_frames, sample_rate, hop_length)
    # amt-tools adds one frame time past the end to bound offsets
    bounded_times = np.append(times, times[-1] + hop_length / sample_rate)

    first = np.searchsorted(bounded_times, onsets, side="right") - 1
    last = np.searchsorted(bounded_times, offsets, side="right") - 1

    first = np.clip(first, 0, num_frames - 1)
    last = np.minimum(last, num_frames - 1)
    return first, last


def notes_to_stacked_multi_pitch(notes, num_frames, sample_rate=TABCNN_SAMPLE_RATE,
                                 hop_length=TABCNN_HOP_LENGTH, tuning=STANDARD_TUNING,
                                 num_frets=TABCNN_NUM_FRETS):
    """
    Activity of every pitch on every string in every frame.

    Returns
    -------
    stacked : bool ndarray (S x P x T)
    stats : FramingStats
    """
    num_strings = len(tuning)
    low, high = pitch_range(tuning, num_frets)
    stacked = np.zeros((num_strings, high - low + 1, num_frames), dtype=bool)
    stats = FramingStats(num_notes=len(notes))

    if num_frames == 0 or len(notes) == 0:
        return stacked, stats

    onsets = notes["onset"].to_numpy(dtype=float)
    offsets = notes["offset"].to_numpy(dtype=float)
    pitches = np.round(notes["pitch"].to_numpy(dtype=float)).astype(int)
    strings = notes["string"].to_numpy(dtype=int)

    # Same filtering as amt-tools' filter_notes
    end_of_grid = num_frames * hop_length / sample_rate
    in_range = (pitches >= low) & (pitches <= high)
    on_grid = (onsets <= end_of_grid) & (offsets >= 0)
    keep = in_range & on_grid

    stats.out_of_range_pitch = int((~in_range).sum())
    stats.outside_grid = int((in_range & ~on_grid).sum())
    stats.notes_used = int(keep.sum())

    first, last = note_frame_bounds(onsets[keep], offsets[keep], num_frames, sample_rate, hop_length)
    for string, pitch, start, stop in zip(strings[keep], pitches[keep] - low, first, last):
        stacked[string, pitch, start:stop + 1] = True

    return stacked, stats


def stacked_multi_pitch_to_tablature(stacked, tuning=STANDARD_TUNING, num_frets=TABCNN_NUM_FRETS):
    """
    Fret of each string in each frame, -1 when the string is silent.
    Only frets 0..num_frets are representable; when several are active, the lowest wins.

    Returns
    -------
    tablature : int64 ndarray (S x T)
    overlapping_frames : number of (string, frame) pairs with more than one active fret
    """
    low = min(tuning)
    num_strings, _, num_frames = stacked.shape
    tablature = np.full((num_strings, num_frames), -1, dtype=np.int64)
    overlapping_frames = 0

    for string in range(num_strings):
        open_index = tuning[string] - low
        frets = stacked[string, open_index:open_index + num_frets + 1]  # (num_frets + 1) x T
        active = frets.any(axis=0)
        tablature[string, active] = np.argmax(frets[:, active], axis=0)
        overlapping_frames += int((frets.sum(axis=0) > 1).sum())

    return tablature, overlapping_frames


def frame_notes(notes, num_frames, sample_rate=TABCNN_SAMPLE_RATE, hop_length=TABCNN_HOP_LENGTH,
                tuning=STANDARD_TUNING, num_frets=TABCNN_NUM_FRETS):
    """
    Frame-level targets for one track.

    Parameters
    ----------
    notes : canonical notes table (schema.NOTE_COLUMNS)
    num_frames : number of frames of the track's features (see num_frames_for_audio_file)
    sample_rate, hop_length : the feature extractor's settings
    tuning : MIDI pitches of the open strings (TrackMeta.tuning)
    num_frets : highest fret the model can output

    Returns
    -------
    targets : dict with
        "times"        float64 (T)     start time of each frame
        "tablature"    int64   (S x T) fret per string per frame, -1 = silent
        "multi_pitch"  float64 (P x T) 1 where a pitch is active on any string
    stats : FramingStats
    """
    stacked, stats = notes_to_stacked_multi_pitch(notes, num_frames, sample_rate, hop_length, tuning, num_frets)
    tablature, stats.overlapping_frames = stacked_multi_pitch_to_tablature(stacked, tuning, num_frets)

    # Notes on a string above num_frets are in multi_pitch but not in the tablature
    if len(notes):
        low = min(tuning)
        pitches = np.round(notes["pitch"].to_numpy(dtype=float)).astype(int)
        frets = pitches - np.asarray(tuning)[notes["string"].to_numpy(dtype=int)]
        in_range = (pitches >= low) & (pitches <= pitch_range(tuning, num_frets)[1])
        stats.above_num_frets = int(((frets > num_frets) & in_range).sum())

    targets = {
        "times": frame_times(num_frames, sample_rate, hop_length),
        "tablature": tablature,
        "multi_pitch": stacked.any(axis=0).astype(np.float64),
    }
    return targets, stats


def frame_track(meta, sample_rate=TABCNN_SAMPLE_RATE, hop_length=TABCNN_HOP_LENGTH,
                num_frets=TABCNN_NUM_FRETS):
    """
    Frame-level targets for a track of a canonical dataset (one TrackMeta of a manifest).
    The number of frames is taken from the audio file, so it matches the features.
    """
    notes = load_notes(meta.notes_path)
    num_frames = num_frames_for_audio_file(meta.audio_path, sample_rate, hop_length)
    return frame_notes(notes, num_frames, sample_rate, hop_length, meta.tuning, num_frets)
