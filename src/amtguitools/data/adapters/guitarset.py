"""
GuitarSet adapter: converts raw GuitarSet into the canonical format (see ../schema.py).

Based on amt_tools/datasets/guitarset.py by Frank Cwitkowitz
(https://github.com/cwitkowitz/amt-tools). See THIRD_PARTY_LICENSES.

What it produces in out_dir:
    notes/<track>.csv      canonical notes (onset, offset, pitch, string, fret)
    contours/<track>.npz   pitch contours per string (for FretNet)
    manifest.csv           one row per track

What it does NOT do (on purpose): loading/resampling audio, computing frame times,
building frame-level tablature, caching features. Those are model-side steps.

Usage:
    python -m amtguitools.data.adapters.guitarset prepare <raw_dir> <out_dir>
    python -m amtguitools.data.adapters.guitarset splits <out_dir>/manifest.csv <split_file.json>
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

# How the standard would look
from ..schema import (
    NOTE_COLUMNS,
    NUM_STRINGS,
    STANDARD_TUNING,
    TrackMeta,
    compute_frets,
    save_manifest,
    save_notes,
    canon_check,
)

DATASET_NAME = "guitarset"

# Audio source used by amt-tools (keep the same to match its features)
AUDIO_DIR = "audio_mono-mic"
AUDIO_SUFFIX = "_mic.wav"
ANNOTATION_DIR = "annotation"

# Track names look like "00_BN1-129-Eb_comp": player, style + progression, tempo, key, mode
TRACK_NAME_PATTERN = re.compile(
    r"^(?P<player>\d{2})_(?P<style>[A-Za-z]+)(?P<progression>\d+)-(?P<tempo>\d+)-(?P<key>[A-G][b#]?)_(?P<mode>comp|solo)$"
)

PLAYERS = ["00", "01", "02", "03", "04", "05"]


# ----------------------------------------------------------------------------
# Download (unchanged from amt-tools, uses mirdata)
# ----------------------------------------------------------------------------

def download(save_dir):
    """Download GuitarSet to save_dir using mirdata."""
    from mirdata.datasets import guitarset  # imported here so mirdata is only needed for downloading

    guitarset.Dataset(data_home=save_dir).download(force_overwrite=True, cleanup=True)


# ----------------------------------------------------------------------------
# Track listing and metadata
# ----------------------------------------------------------------------------

def list_tracks(raw_dir):
    """Names of all tracks, from the JAMS files (other files are ignored), sorted."""
    jams_dir = os.path.join(raw_dir, ANNOTATION_DIR)
    names = [os.path.splitext(f)[0] for f in os.listdir(jams_dir) if f.endswith(".jams")]
    return sorted(names)


def parse_track_name(name):
    """Metadata encoded in the track name. Falls back to the player code only."""
    match = TRACK_NAME_PATTERN.match(name)
    if match is None:
        return {"player": name[:2]}
    return match.groupdict()


# ----------------------------------------------------------------------------
# JAMS parsing (JAMS files are plain JSON, so no extra library is needed)
# ----------------------------------------------------------------------------

def _observations(data):
    """
    Return a JAMS annotation's data as a list of dicts with keys time, duration, value.
    JAMS stores data either as a list of observations or as a dict of columns.
    """
    if isinstance(data, dict):
        return [
            {"time": t, "duration": d, "value": v}
            for t, d, v in zip(data["time"], data["duration"], data["value"])
        ]
    return data


def _annotations_by_string(jam, namespace):
    """
    The annotations of a namespace, as a list of (string_index, annotation).
    The string index comes from annotation_metadata.data_source (0 = low E);
    if that is missing, the order of appearance in the file is used.
    """
    annotations = [a for a in jam["annotations"] if a["namespace"] == namespace]
    result = []
    for order, annotation in enumerate(annotations):
        source = str(annotation.get("annotation_metadata", {}).get("data_source", ""))
        string = int(source) if source.isdigit() else order
        result.append((string, annotation))
    return result


def notes_from_jams(jam, tuning=STANDARD_TUNING):
    """
    Canonical notes table from a GuitarSet JAMS dict.

    Returns
    -------
    notes : DataFrame with NOTE_COLUMNS, sorted by onset then string
    num_dropped : number of notes dropped because their fret would be negative
    """
    rows = []
    for string, annotation in _annotations_by_string(jam, "note_midi"):
        for obs in _observations(annotation["data"]):
            onset = float(obs["time"])
            rows.append((onset, onset + float(obs["duration"]), float(obs["value"]), string))

    notes = pd.DataFrame(rows, columns=["onset", "offset", "pitch", "string"])
    notes["string"] = notes["string"].astype(int)
    notes["fret"] = compute_frets(notes["pitch"].to_numpy(), notes["string"].to_numpy(), tuning)

    # A note slightly flatter than its open string would get fret -1: drop it and report it
    negative = notes["fret"] < 0
    num_dropped = int(negative.sum())
    notes = notes[~negative]

    notes = notes.sort_values(["onset", "string"]).reset_index(drop=True)
    return notes[NOTE_COLUMNS], num_dropped


def contours_from_jams(jam):
    """
    Pitch contours per string from a GuitarSet JAMS dict, as arrays ready for np.savez:
        s<k>_times, s<k>_freqs (Hz), s<k>_voiced   for k = 0..5 (0 = low E)
    """
    arrays = {}
    for string in range(NUM_STRINGS):
        arrays[f"s{string}_times"] = np.zeros(0)
        arrays[f"s{string}_freqs"] = np.zeros(0)
        arrays[f"s{string}_voiced"] = np.zeros(0, dtype=bool)

    for string, annotation in _annotations_by_string(jam, "pitch_contour"):
        observations = _observations(annotation["data"])
        arrays[f"s{string}_times"] = np.array([float(o["time"]) for o in observations])
        arrays[f"s{string}_freqs"] = np.array([float(o["value"]["frequency"]) for o in observations])
        arrays[f"s{string}_voiced"] = np.array([bool(o["value"].get("voiced", True)) for o in observations])

    return arrays


# ----------------------------------------------------------------------------
# Prepare: raw GuitarSet -> canonical format
# ----------------------------------------------------------------------------

def prepare_guitarset(raw_dir, out_dir):
    """
    Convert raw GuitarSet (as downloaded) into the canonical format in out_dir.
    Safe to rerun: every output file is rewritten.
    """
    raw_dir = os.path.abspath(raw_dir)
    out_dir = os.path.abspath(out_dir)
    notes_dir = os.path.join(out_dir, "notes")
    contours_dir = os.path.join(out_dir, "contours")
    os.makedirs(notes_dir, exist_ok=True)
    os.makedirs(contours_dir, exist_ok=True)

    metas, extras = [], []
    total_dropped = 0

    for name in list_tracks(raw_dir):
        jams_path = os.path.join(raw_dir, ANNOTATION_DIR, name + ".jams")
        audio_path = os.path.join(raw_dir, AUDIO_DIR, name + AUDIO_SUFFIX)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Missing audio for {name}: {audio_path}")

        with open(jams_path) as f:
            jam = json.load(f)

        info = parse_track_name(name)
        meta = TrackMeta(
            track_id=f"{DATASET_NAME}/{name}",
            dataset=DATASET_NAME,
            audio_path=audio_path,
            notes_path=os.path.join(notes_dir, name + ".csv"),
            duration=float(jam["file_metadata"]["duration"]),
            group=info["player"],
            tuning=STANDARD_TUNING,
            contours_path=os.path.join(contours_dir, name + ".npz"),
        )

        notes, num_dropped = notes_from_jams(jam, meta.tuning)
        total_dropped += num_dropped
        canon_check(notes, meta)

        save_notes(notes, meta.notes_path)
        np.savez(meta.contours_path, **contours_from_jams(jam))

        metas.append(meta)
        extras.append({k: v for k, v in info.items() if k != "player"})

    manifest_path = os.path.join(out_dir, "manifest.csv")
    save_manifest(metas, manifest_path, extras)

    print(f"Prepared {len(metas)} tracks into {out_dir}")
    if total_dropped:
        print(f"Dropped {total_dropped} notes with negative fret (pitch below the open string)")
    return manifest_path


# ----------------------------------------------------------------------------
# Splits: standard 6-fold leave-one-player-out protocol
# ----------------------------------------------------------------------------

def write_6fold_player_splits(manifest_path, split_path):
    """
    Write the split file for the standard GuitarSet protocol: fold k tests on player k
    and trains on the other five players. No validation set (as in TabCNN).
    """
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    folds = {}
    for fold, player in enumerate(PLAYERS):
        is_test = manifest["group"] == player
        folds[str(fold)] = {
            "test_group": player,
            "train": sorted(manifest.loc[~is_test, "track_id"]),
            "test": sorted(manifest.loc[is_test, "track_id"]),
        }

    os.makedirs(os.path.dirname(os.path.abspath(split_path)), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"protocol": "guitarset_6fold_player", "folds": folds}, f, indent=2)
    print(f"Wrote {len(folds)} folds to {split_path}")


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GuitarSet adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download raw GuitarSet")
    p.add_argument("save_dir")

    p = sub.add_parser("prepare", help="convert raw GuitarSet to the canonical format")
    p.add_argument("raw_dir")
    p.add_argument("out_dir")

    p = sub.add_parser("splits", help="write the 6-fold leave-one-player-out split file")
    p.add_argument("manifest_path")
    p.add_argument("split_path")

    args = parser.parse_args()
    if args.command == "download":
        download(args.save_dir)
    elif args.command == "prepare":
        prepare_guitarset(args.raw_dir, args.out_dir)
    else:
        write_6fold_player_splits(args.manifest_path, args.split_path)


if __name__ == "__main__":
    main()