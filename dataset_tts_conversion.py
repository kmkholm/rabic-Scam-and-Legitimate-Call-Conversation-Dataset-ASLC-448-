#!/usr/bin/env python3
"""
=============================================================================
 Arabic Scam Dataset — Text-to-Speech (Google Colab Compatible)
 Save to: /content/drive/MyDrive/audio_dataset/
=============================================================================
"""

import os, asyncio, time, warnings, random, shutil
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
from pydub import AudioSegment

warnings.filterwarnings("ignore")

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️ CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH    = "/content/arabic_scam_dataset_complete.xlsx"
OUTPUT_DIR      = "/content/drive/MyDrive/audio_dataset2"
ENGINE          = "edge"       # "edge" or "gtts"
SAMPLE_RATE     = 16000
SILENCE_MS      = 600
PHONE_FILTER    = True
ADD_NOISE       = True
NOISE_DB        = -35
LIMIT           = None         # None=all, or 5 for testing
FILTER_DIALECT  = None         # None=all, or "Iraqi"

# ─────────────────────────────────────────────────────────────────────────────
# Voice mapping
# ─────────────────────────────────────────────────────────────────────────────
EDGE_VOICES = {
    "Egyptian":   ("ar-EG-ShakirNeural",  "ar-EG-SalmaNeural"),
    "Gulf":       ("ar-AE-HamdanNeural",  "ar-AE-FatimaNeural"),
    "Saudi":      ("ar-SA-HamedNeural",   "ar-SA-ZariyahNeural"),
    "Jordanian":  ("ar-JO-TaimNeural",    "ar-JO-SanaNeural"),
    "Iraqi":      ("ar-IQ-BasselNeural",  "ar-IQ-RanaNeural"),
    "Syrian":     ("ar-SY-AmanyNeural",   "ar-SY-LaithNeural"),
    "Sudanese":   ("ar-SA-HamedNeural",   "ar-EG-SalmaNeural"),
    "MSA":        ("ar-SA-HamedNeural",   "ar-EG-SalmaNeural"),
}

TURN_ORDER = [
    ("caller_turn_1",   "caller"),
    ("receiver_turn_1", "receiver"),
    ("caller_turn_2",   "caller"),
    ("receiver_turn_2", "receiver"),
    ("caller_turn_3",   "caller"),
]

# ─────────────────────────────────────────────────────────────────────────────
# TTS Engines
# ─────────────────────────────────────────────────────────────────────────────
class EdgeTTSEngine:
    def __init__(self):
        import edge_tts
        self.edge_tts = edge_tts

    def synthesize(self, text, voice, output_path):
        async def _run():
            communicate = self.edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_run())

    def get_voices(self, dialect):
        return EDGE_VOICES.get(dialect, EDGE_VOICES["MSA"])


class GTTSEngine:
    def __init__(self):
        from gtts import gTTS
        self.gTTS = gTTS

    def synthesize(self, text, voice, output_path):
        tts = self.gTTS(text=text, lang="ar", slow=False)
        mp3_path = output_path.replace(".wav", ".mp3")
        tts.save(mp3_path)
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(output_path, format="wav")
        os.remove(mp3_path)

    def get_voices(self, dialect):
        return ("ar", "ar")


# ─────────────────────────────────────────────────────────────────────────────
# Audio Processing
# ─────────────────────────────────────────────────────────────────────────────
def bandpass_filter(audio_segment, lowcut=300, highcut=3400):
    """Telephone band-pass filter using scipy."""
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float64)
    sr = audio_segment.frame_rate

    nyquist = sr / 2.0
    low = max(lowcut / nyquist, 0.01)
    high = min(highcut / nyquist, 0.99)

    b, a = butter(4, [low, high], btype='band')
    filtered = lfilter(b, a, samples)

    filtered = np.clip(filtered, -32768, 32767).astype(np.int16)
    return audio_segment._spawn(filtered.tobytes())


def add_noise(audio_segment, noise_db=-35):
    """Add light white noise."""
    n_samples = int(len(audio_segment) * audio_segment.frame_rate / 1000)
    noise_samples = np.random.normal(0, 300, n_samples).astype(np.int16)
    noise = AudioSegment(
        noise_samples.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=2,
        channels=1
    )
    noise = noise + noise_db
    if len(noise) < len(audio_segment):
        noise = noise * (len(audio_segment) // len(noise) + 1)
    noise = noise[:len(audio_segment)]
    return audio_segment.overlay(noise)


def normalize(audio_segment, target=-20):
    """Normalize volume."""
    if audio_segment.dBFS == float('-inf'):
        return audio_segment
    change = target - audio_segment.dBFS
    return audio_segment.apply_gain(change)


def vary_speed(audio_segment):
    """Slight speed variation ±5%."""
    speed = random.uniform(0.95, 1.05)
    new_rate = int(audio_segment.frame_rate * speed)
    return audio_segment._spawn(
        audio_segment.raw_data,
        overrides={"frame_rate": new_rate}
    ).set_frame_rate(audio_segment.frame_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Convert one conversation
# ─────────────────────────────────────────────────────────────────────────────
def convert_one(row, engine, temp_dir):
    conv_id = row['conversation_id']
    dialect = row['dialect']
    caller_voice, receiver_voice = engine.get_voices(dialect)
    voices = {"caller": caller_voice, "receiver": receiver_voice}

    segments = []
    for col, role in TURN_ORDER:
        text = str(row[col]).strip()
        if not text or text == "nan":
            continue
        temp_file = os.path.join(temp_dir, f"{conv_id}_{col}.wav")
        try:
            engine.synthesize(text, voices[role], temp_file)
            seg = AudioSegment.from_file(temp_file)
            seg = vary_speed(seg)
            segments.append(seg)
        except Exception as e:
            print(f"    ⚠ {conv_id}/{col}: {e}")

    if not segments:
        return None

    # Merge with silence
    silence = AudioSegment.silent(duration=SILENCE_MS)
    merged = segments[0]
    for s in segments[1:]:
        merged = merged + silence + s

    # Normalize
    merged = normalize(merged)

    # Phone filter (scipy)
    if PHONE_FILTER:
        merged = bandpass_filter(merged, 300, 3400)

    # Noise
    if ADD_NOISE:
        merged = add_noise(merged, NOISE_DB)

    # 16kHz mono
    merged = merged.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)

    # Save
    label_dir = os.path.join(OUTPUT_DIR, row['label'])
    os.makedirs(label_dir, exist_ok=True)
    out_path = os.path.join(label_dir, f"{conv_id}.wav")
    merged.export(out_path, format="wav")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# ▶️ MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run():
    print(f"✓ Loading: {DATASET_PATH}")
    df = pd.read_excel(DATASET_PATH)
    print(f"  Total: {len(df)} conversations")

    if FILTER_DIALECT:
        df = df[df['dialect'] == FILTER_DIALECT]
        print(f"  Filtered to '{FILTER_DIALECT}': {len(df)}")
    if LIMIT:
        df = df.head(LIMIT)
        print(f"  Limited to: {len(df)}")

    engine = EdgeTTSEngine() if ENGINE == "edge" else GTTSEngine()
    print(f"✓ Engine: {ENGINE}")

    temp_dir = os.path.join(OUTPUT_DIR, "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Converting {len(df)} conversations → WAV")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    results = []
    t0 = time.time()
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        cid = row['conversation_id']
        try:
            path = convert_one(row, engine, temp_dir)
            if path:
                dur = len(AudioSegment.from_file(path)) / 1000
                results.append({"conversation_id": cid, "audio_path": path,
                                "duration_sec": round(dur, 1), "label": row['label'],
                                "dialect": row['dialect']})
                status = f"✓ {dur:.1f}s"
            else:
                status = "✗ empty"
        except Exception as e:
            status = f"✗ {e}"

        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (total - i - 1)
        print(f"  [{i+1:3d}/{total}] {cid} | {row['dialect']:10s} | {row['label']:8s} | {status} | ETA:{eta:.0f}s")

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Save metadata
    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(OUTPUT_DIR, "audio_metadata.csv"), index=False)

    # Update Excel
    amap = {r['conversation_id']: r['audio_path'] for r in results}
    dmap = {r['conversation_id']: r['duration_sec'] for r in results}
    df['audio_path'] = df['conversation_id'].map(amap)
    df['audio_duration'] = df['conversation_id'].map(dmap)
    df.to_excel(os.path.join(OUTPUT_DIR, "dataset_with_audio.xlsx"), index=False)

    success = sum(1 for r in results if r.get('audio_path'))
    total_dur = sum(r['duration_sec'] for r in results)
    elapsed = time.time() - t0

    print(f"""
{'='*60}
  ✅ DONE!
{'='*60}
  Files      : {success}/{total}
  Total audio: {total_dur:.0f}s ({total_dur/60:.1f} min)
  Time       : {elapsed:.0f}s

  {OUTPUT_DIR}/
  ├── scam/         ({sum(1 for r in results if r['label']=='scam')} files)
  ├── not_scam/     ({sum(1 for r in results if r['label']=='not_scam')} files)
  ├── audio_metadata.csv
  └── dataset_with_audio.xlsx
{'='*60}
""")
    return rdf

# ── RUN ──
if __name__ == "__main__":
    results = run()
