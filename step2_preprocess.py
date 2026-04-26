# ============================================================
# STEP 2: PREPROCESSING (Upgraded Version)
# UPGRADE 1: Each song → multiple 3-second segments (~10x more data)
# UPGRADE 2: Simple audio augmentation (time shift + noise)
# ============================================================

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# ---- Configuration ----
DATA_PATH      = "data/genres_original"
SAVE_PATH      = "data/processed"
SAMPLE_RATE    = 22050
SEGMENT_SECS   = 3
SEGMENT_LENGTH = SEGMENT_SECS * SAMPLE_RATE
N_MELS         = 128
HOP_LENGTH     = 512
N_FFT          = 2048

GENRES = ['blues', 'classical', 'country', 'disco',
          'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

os.makedirs(SAVE_PATH, exist_ok=True)

def split_into_segments(file_path):
    try:
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        segments = []
        num_segments = len(waveform) // SEGMENT_LENGTH
        for i in range(num_segments):
            start   = i * SEGMENT_LENGTH
            end     = start + SEGMENT_LENGTH
            segment = waveform[start:end]
            if len(segment) < SEGMENT_LENGTH:
                continue
            segments.append(segment)
        return segments
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def augment_segment(segment):
    if np.random.rand() < 0.5:
        shift   = np.random.randint(-2000, 2000)
        segment = np.roll(segment, shift)
    if np.random.rand() < 0.3:
        noise   = 0.005 * np.random.randn(len(segment))
        segment = segment + noise
    return segment

def segment_to_melspectrogram(segment):
    try:
        mel    = librosa.feature.melspectrogram(
            y=segment, sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    except:
        return None

def segment_to_mfcc(segment, n_mfcc=13):
    try:
        mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
        return mfcc.T
    except:
        return None

def process_dataset():
    print("=" * 60)
    print("PREPROCESSING DATASET (UPGRADED VERSION)")
    print("=" * 60)
    print(f"Segment length   : {SEGMENT_SECS} seconds")
    print(f"Expected samples : ~9000 (10x more than before)")
    print()

    spectrograms = []
    mfccs        = []
    labels       = []
    failed       = 0

    for genre_idx, genre in enumerate(GENRES):
        genre_path = os.path.join(DATA_PATH, genre)
        if not os.path.exists(genre_path):
            print(f"⚠ Not found: {genre_path}")
            continue

        files = [f for f in os.listdir(genre_path)
                 if f.endswith(('.wav', '.mp3', '.au'))]

        print(f"[{genre_idx+1}/10] {genre.upper()} ({len(files)} songs)")

        for fname in tqdm(files, desc=f"  {genre}", leave=False):
            path     = os.path.join(genre_path, fname)
            segments = split_into_segments(path)

            if not segments:
                failed += 1
                continue

            for seg_idx, segment in enumerate(segments):
                aug_segment = augment_segment(segment)
                mel         = segment_to_melspectrogram(aug_segment)
                mfcc        = segment_to_mfcc(aug_segment)

                if mel is not None and mfcc is not None:
                    spectrograms.append(mel)
                    mfccs.append(mfcc)
                    labels.append(genre_idx)

    total = len(spectrograms)
    print(f"\n✓ Total segments : {total}")
    print(f"✗ Failed files   : {failed}")
    print(f"📈 Data increase : ~{total // 999}x more samples than before")

    print("\nConverting to arrays...")
    spectrograms = np.array(spectrograms, dtype=np.float32)
    labels       = np.array(labels, dtype=np.int64)
    print(f"Spectrogram shape : {spectrograms.shape}")

    mean = spectrograms.mean()
    std  = spectrograms.std()
    spectrograms = (spectrograms - mean) / std
    print(f"Normalized        : mean={mean:.2f}, std={std:.2f}")

    max_len      = max(m.shape[0] for m in mfccs)
    mfccs_padded = np.array([
        np.pad(m, ((0, max_len - m.shape[0]), (0, 0))) for m in mfccs
    ], dtype=np.float32)

    print("\nSaving...")
    np.save(os.path.join(SAVE_PATH, 'spectrograms.npy'), spectrograms)
    np.save(os.path.join(SAVE_PATH, 'labels.npy'),       labels)
    np.save(os.path.join(SAVE_PATH, 'mfccs.npy'),        mfccs_padded)
    np.save(os.path.join(SAVE_PATH, 'norm_stats.npy'),   np.array([mean, std]))

    meta = {
        'genres':            GENRES,
        'total_samples':     total,
        'segment_seconds':   SEGMENT_SECS,
        'spectrogram_shape': list(spectrograms.shape),
        'mfcc_shape':        list(mfccs_padded.shape),
        'norm_mean':         float(mean),
        'norm_std':          float(std),
    }
    with open(os.path.join(SAVE_PATH, 'metadata.json'), 'w') as fp:
        json.dump(meta, fp, indent=2)

    print("✓ All saved to data/processed/")
    return spectrograms, mfccs_padded, labels

def visualize_sample():
    for genre in ['rock', 'classical', 'hiphop']:
        genre_path = os.path.join(DATA_PATH, genre)
        if not os.path.exists(genre_path):
            continue
        files = [f for f in os.listdir(genre_path)
                 if f.endswith(('.wav', '.au', '.mp3'))]
        if not files:
            continue
        path     = os.path.join(genre_path, files[0])
        waveform, sr = librosa.load(path, sr=SAMPLE_RATE)
        segment  = waveform[:SEGMENT_LENGTH]
        mel      = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS)
        mel_db   = librosa.power_to_db(mel, ref=np.max)
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        fig.suptitle(f"Sample: {genre.upper()} — First 3-second segment", fontsize=14)
        librosa.display.waveshow(segment, sr=sr, ax=axes[0])
        axes[0].set_title("Waveform (3 seconds)")
        img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel',
                                        sr=sr, ax=axes[1])
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        axes[1].set_title("Mel Spectrogram")
        plt.tight_layout()
        out = os.path.join(SAVE_PATH, 'sample_segment.png')
        plt.savefig(out, dpi=130, bbox_inches='tight')
        plt.show()
        print(f"Saved: {out}")
        break

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print("⚠ Dataset not found at:", DATA_PATH)
    else:
        spectrograms, mfccs, labels = process_dataset()
        print("\nGenerating sample visualization...")
        visualize_sample()
        print("\n" + "=" * 60)
        print("✓ PREPROCESSING COMPLETE (UPGRADED)")
        print("=" * 60)
        print(f"\nTotal training samples : {len(labels)}")
        print("(vs 999 before the upgrade)")
        print("\nNEXT STEP: python step3_train.py")
