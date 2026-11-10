"""
Canonical data format shared by every dataset adapter.

Every adapter converts its raw dataset into this format. Everything downstream
(framing, features, training, evaluation) only reads this format.

-----------
- Times are in seconds.
- Pitch is a MIDI number stored as a float (bends and vibrato are kept).
- Strings are indexed 0..5, with 0 = lowest string (low E in standard tuning).
- Tuning is stored per track as the MIDI pitches of the open strings.
- Fret is always round(pitch) - tuning[string].

"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

NUM_STRINGS = 6

# MIDI pitches of the open strings, string 0 = low E
STANDARD_TUNING = (40, 45, 50, 55, 59, 64)

# Columns of a canonical notes table, in this order
NOTE_COLUMNS = ["onset", "offset", "pitch", "string", "fret"]


@dataclass
class TrackMeta:
    """One row of a dataset manifest."""

    track_id: str        
    dataset: str        
    audio_path: str      
    notes_path: str  
    duration: float      
    group: str          
    tuning: tuple = STANDARD_TUNING
    contours_path: str = ""  # optional path to pitch contours (NPZ), "" if none


def canon_check(notes, meta):
    """
    Check that a notes table follows the canonical format. Raises ValueError otherwise.
    """
    if list(notes.columns) != NOTE_COLUMNS:
        raise ValueError(f"{meta.track_id}: columns {list(notes.columns)} != {NOTE_COLUMNS}")

    if len(meta.tuning) != NUM_STRINGS:
        raise ValueError(f"{meta.track_id}: tuning must have {NUM_STRINGS} values")

    if len(notes) == 0:
        return

    if not (notes["onset"] >= 0).all():
        raise ValueError(f"{meta.track_id}: negative onsets")

    if not notes["string"].between(0, NUM_STRINGS - 1).all():
        raise ValueError(f"{meta.track_id}: string index out of range")

    if not (notes["fret"] >= 0).all():
        raise ValueError(f"{meta.track_id}: negative frets")

    expected = compute_frets(notes["pitch"].to_numpy(), notes["string"].to_numpy(), meta.tuning)
    if not (notes["fret"].to_numpy() == expected).all():
        raise ValueError(f"{meta.track_id}: fret != round(pitch) - tuning[string]")


def compute_frets(pitches, strings, tuning):
    """Fret of each note: round(pitch) - tuning[string]."""
    open_pitches = np.asarray(tuning)[np.asarray(strings, dtype=int)]
    return np.round(np.asarray(pitches, dtype=float)).astype(int) - open_pitches


# ----------------------------------------------------------------------------
# Reading and writing
# ----------------------------------------------------------------------------

def save_notes(notes, path):
    """Write a canonical notes table to CSV."""
    notes[NOTE_COLUMNS].to_csv(path, index=False)


def load_notes(path):
    """Read a canonical notes table from CSV."""
    notes = pd.read_csv(path)
    notes["string"] = notes["string"].astype(int)
    notes["fret"] = notes["fret"].astype(int)
    return notes[NOTE_COLUMNS]


def save_manifest(metas, path, extras=None):
    """
    Write a list of TrackMeta to CSV. Tuning is stored as space-separated MIDI pitches.

    extras : optional list of dicts (one per track) with dataset-specific columns
             (e.g. style, mode). They are saved for analysis but ignored by load_manifest().
    """
    rows = []
    for i, meta in enumerate(metas):
        row = asdict(meta)
        row["tuning"] = " ".join(str(p) for p in meta.tuning)
        if extras is not None:
            row.update(extras[i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_manifest(path):
    """Read a manifest CSV into a list of TrackMeta (extra columns are ignored)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    fields = TrackMeta.__dataclass_fields__.keys()
    metas = []
    for row in df.to_dict("records"):
        row = {key: value for key, value in row.items() if key in fields}
        row["tuning"] = tuple(int(p) for p in row["tuning"].split())
        row["duration"] = float(row["duration"])
        metas.append(TrackMeta(**row))
    return metas