import librosa
import numpy as np
from amtguitools.data.framing import TABCNN_SAMPLE_RATE, TABCNN_HOP_LENGTH

# SAMPLE_RATE = 22050
# HOP_LENGTH = 512
SAMPLE_RATE = TABCNN_SAMPLE_RATE
HOP_LENGTH = TABCNN_HOP_LENGTH
N_BINS = 192
BINS_PER_OCTAVE = 24

def compute_cqt(audio_path):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, res_type="kaiser_best")
    if np.sqrt(np.mean(audio ** 2))>0: 
        audio = audio / np.sqrt(np.mean(audio ** 2))          # RMS normalization
    cqt = librosa.cqt(audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                      n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
    db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)  # -80..0 dB
    return db / 80 + 1                                     # 0..1, shape (192, T)