#!/usr/bin/env python3
"""
=============================================================================
 Build Complete Arabic Scam Audio Dataset
 Combines: Audio WAV files + Text + Features + Train/Val/Test Splits
=============================================================================
"""

import os, warnings, json, time
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ⚙️ CONFIG
AUDIO_DIR       = "/content/drive/MyDrive/audio_dataset2"
TEXT_DATASET    = "/content/arabic_scam_dataset_complete.xlsx"
OUTPUT_DIR      = "/content/drive/MyDrive/final_dataset"
SAMPLE_RATE     = 16000
TEST_SIZE       = 0.20
VAL_SIZE        = 0.10
SEED            = 42
EXTRACT_FEATURES = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SCAN & MATCH AUDIO FILES
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 1: Scanning Audio Files & Matching with Text Data")
print("=" * 65)

df_text = pd.read_excel(TEXT_DATASET)
print(f"\n✓ Text dataset: {len(df_text)} conversations")

audio_files = {}
for label_folder in ["scam", "not_scam"]:
    folder = os.path.join(AUDIO_DIR, label_folder)
    if not os.path.exists(folder):
        continue
    for f in os.listdir(folder):
        if f.endswith(".wav"):
            conv_id = f.replace(".wav", "")
            audio_files[conv_id] = {
                "audio_path": os.path.join(folder, f),
                "audio_label_folder": label_folder
            }

print(f"✓ Audio files found: {len(audio_files)}")

df_text['audio_path'] = df_text['conversation_id'].map(
    lambda x: audio_files[x]['audio_path'] if x in audio_files else None
)
df_text['has_audio'] = df_text['audio_path'].notna()

matched = df_text['has_audio'].sum()
print(f"\n✓ Matched: {matched}/{len(df_text)} ({100*matched/len(df_text):.1f}%)")

df = df_text[df_text['has_audio']].copy().reset_index(drop=True)
print(f"\n✓ Final dataset: {len(df)} conversations with audio")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACT AUDIO FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  STEP 2: Extracting Audio Features")
print(f"{'=' * 65}\n")

def extract_audio_features(audio_path, sr=SAMPLE_RATE):
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        return None

    duration = len(y) / sr

    features = {
        "duration_sec": round(duration, 2),
        "n_samples": len(y),
        "rms_energy": float(np.sqrt(np.mean(y**2))),
        "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
    }

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features["mel_mean"] = float(np.mean(mel_db))
    features["mel_std"] = float(np.std(mel_db))

    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(spec_cent))

    return features

if EXTRACT_FEATURES:
    print("Extracting features from audio files...\n")
    all_features = []
    t0 = time.time()

    for i, row in df.iterrows():
        feats = extract_audio_features(row['audio_path'])
        if feats:
            feats['conversation_id'] = row['conversation_id']
            all_features.append(feats)

        if (i + 1) % 50 == 0 or i == len(df) - 1:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (len(df) - i - 1)
            print(f"  [{i+1:3d}/{len(df)}] extracted | ETA: {eta:.0f}s")

    df_features = pd.DataFrame(all_features)
    print(f"\n✓ Extracted {len(df_features.columns)-1} features from {len(df_features)} files")

    df = df.merge(df_features, on='conversation_id', how='left')
    print(f"✓ Merged: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  STEP 3: Creating Train / Val / Test Splits")
print(f"{'=' * 65}\n")

df_trainval, df_test = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED, stratify=df['label']
)
df_train, df_val = train_test_split(
    df_trainval, test_size=VAL_SIZE / (1 - TEST_SIZE),
    random_state=SEED, stratify=df_trainval['label']
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train['split'] = 'train'
df_val['split'] = 'val'
df_test['split'] = 'test'

print(f"  Train : {len(df_train)} (scam={sum(df_train['label']=='scam')})")
print(f"  Val   : {len(df_val)}  (scam={sum(df_val['label']=='scam')})")
print(f"  Test  : {len(df_test)}  (scam={sum(df_test['label']=='scam')})")

df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE FINAL DATASET
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  STEP 4: Saving Final Dataset")
print(f"{'=' * 65}\n")

# CSV files per split
df_train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
df_val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
df_test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

# Full dataset
df_full.to_csv(os.path.join(OUTPUT_DIR, "full_dataset.csv"), index=False)
df_full.to_excel(os.path.join(OUTPUT_DIR, "full_dataset.xlsx"), index=False)

# Numpy arrays for ML
if EXTRACT_FEATURES:
    feature_cols = [c for c in df_full.columns if c.startswith(('mfcc_', 'mel_', 'spectral_',
                    'rms_', 'zero_', 'duration_sec'))]

    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        X = split_df[feature_cols].fillna(0).values
        y = (split_df['label'] == 'scam').astype(int).values
        np.save(os.path.join(OUTPUT_DIR, f"X_{split_name}.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, f"y_{split_name}.npy"), y)

    with open(os.path.join(OUTPUT_DIR, "feature_names.json"), 'w') as f:
        json.dump(feature_cols, f, indent=2)

    print(f"✓ Feature arrays saved: {len(feature_cols)} features")

# Dataset info
info = {
    "name": "Arabic Scam Call Audio Dataset",
    "total_samples": len(df_full),
    "splits": {
        "train": len(df_train),
        "val": len(df_val),
        "test": len(df_test)
    },
    "labels": {
        "scam": int(sum(df_full['label'] == 'scam')),
        "not_scam": int(sum(df_full['label'] == 'not_scam'))
    },
    "dialects": df_full['dialect'].value_counts().to_dict(),
    "sample_rate": SAMPLE_RATE,
    "total_duration_sec": round(float(df_full['duration_sec'].sum()), 1),
}

with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False, default=str)

print(f"""
✓ All outputs saved to '{OUTPUT_DIR}/' directory:
  ├── train.csv            ({len(df_train)} rows)
  ├── val.csv              ({len(df_val)} rows)
  ├── test.csv             ({len(df_test)} rows)
  ├── full_dataset.csv
  ├── full_dataset.xlsx
  ├── X_train.npy          (feature matrix)
  ├── y_train.npy          (labels)
  ├── feature_names.json
  └── dataset_info.json

  🚀 Ready for:
     • BERT text classification
     • Audio classification (MFCC features)
     • Multimodal (text + audio) models
""")

print("=" * 65)
print("  ✅ DATASET BUILDER COMPLETE")
print("=" * 65)
