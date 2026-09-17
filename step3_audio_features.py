import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# PATH CONFIGURATION
# ============================================================
METADATA_PATH = PROJECT_ROOT / "processed" / "metadata.csv"
STEP3_DIR = PROJECT_ROOT / "processed" / "step3"
AUDIO_SEGMENTS_DIR = STEP3_DIR / "audio_segments"
AUDIO_METADATA_PATH = STEP3_DIR / "audio_metadata.csv"
AUDIO_FEATURES_PATH = STEP3_DIR / "audio_features.csv"

# Parameters
TARGET_SR = 16000
MONO = True
NORMALIZE_AUDIO = True
N_MFCC = 13
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

STEP3_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df, possible_names):
    normalized_columns = {
        str(col).strip().lower().replace(" ", "_").replace("-", "_"): col
        for col in df.columns
    }
    for name in possible_names:
        norm_name = name.strip().lower().replace(" ", "_").replace("-", "_")
        if norm_name in normalized_columns:
            return normalized_columns[norm_name]
    return None


def resolve_audio_path(audio_path):
    if audio_path is None or pd.isna(audio_path):
        return None
    audio_str = str(audio_path).strip()
    if not audio_str or audio_str.lower() in ["not_available", "none", "nan", "null"]:
        return None

    path = Path(audio_str)
    
    # Danh sách các vị trí kiểm tra đường dẫn
    candidates = [
        path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "dataset" / path,
        PROJECT_ROOT / "dataset" / "E-DAIC" / path,
    ]
    
    # Nếu là đường dẫn tương đối từ dataset gốc
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand

    return candidates[1] # Trả về đường dẫn thử nghiệm nếu không tìm thấy


# ============================================================
# STEP 3A: AUDIO SEGMENTATION
# ============================================================
def load_audio_segment(audio_path, start_time=None, end_time=None):
    audio_file = resolve_audio_path(audio_path)
    if audio_file is None or not audio_file.exists():
        raise FileNotFoundError(f"File không tồn tại: {audio_file}")

    start = float(start_time) if start_time is not None and not pd.isna(start_time) and float(start_time) >= 0 else None
    duration = None
    if end_time is not None and not pd.isna(end_time):
        end = float(end_time)
        if start is not None:
            duration = end - start
            if duration <= 0:
                raise ValueError(f"Timestamp không hợp lệ: {start} -> {end}")

    y, sr = librosa.load(str(audio_file), sr=TARGET_SR, mono=MONO, offset=start, duration=duration)
    if len(y) == 0:
        raise ValueError("Đoạn audio cắt ra bị rỗng (length = 0)")
    return y, sr, audio_file


def run_step_3a():
    print("\n" + "=" * 70 + "\nSTEP 3A - AUDIO SEGMENTATION\n" + "=" * 70)
    if not METADATA_PATH.exists():
        print(f"❌ KHÔNG TÌM THẤY metadata.csv tại: {METADATA_PATH}")
        return False

    df = pd.read_csv(METADATA_PATH)
    part_col = find_column(df, ["participant_id", "participant", "participantid"])
    seg_col = find_column(df, ["segment_id", "segment", "segmentid"])
    audio_col = find_column(df, ["audio_path", "audio", "audio_file"])
    start_col = find_column(df, ["start_time", "start"])
    end_col = find_column(df, ["end_time", "end"])

    results = []
    error_samples = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio"):
        record = row.to_dict()
        p_id = str(row[part_col]) if part_col else "unknown_p"
        s_id = str(row[seg_col]) if seg_col else f"seg_{idx}"
        
        safe_p = p_id.replace("/", "_").replace("\\", "_")
        safe_s = s_id.replace("/", "_").replace("\\", "_")
        out_file = AUDIO_SEGMENTS_DIR / f"{safe_p}_{safe_s}.wav"

        try:
            y, sr, resolved_p = load_audio_segment(row[audio_col], row.get(start_col), row.get(end_col))
            if NORMALIZE_AUDIO and np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            sf.write(str(out_file), y, sr)
            
            rel_out_path = out_file.relative_to(PROJECT_ROOT)
            record.update({
                "processed_audio_path": str(rel_out_path),
                "resolved_original_audio_path": str(resolved_p),
                "audio_duration": float(len(y) / sr),
                "audio_sampling_rate": int(sr),
                "audio_status": "success",
                "audio_error": ""
            })
        except Exception as e:
            if len(error_samples) < 3:
                error_samples.append((idx, str(row.get(audio_col)), str(e)))
            record.update({
                "processed_audio_path": "",
                "resolved_original_audio_path": "",
                "audio_duration": None,
                "audio_sampling_rate": None,
                "audio_status": "error",
                "audio_error": str(e)
            })
        results.append(record)

    out_df = pd.DataFrame(results)
    out_df.to_csv(AUDIO_METADATA_PATH, index=False, encoding="utf-8-sig")
    
    success_cnt = len(out_df[out_df['audio_status']=='success'])
    error_cnt = len(out_df[out_df['audio_status']=='error'])
    
    print(f"\n✅ STEP 3A HOÀN THÀNH: {success_cnt} thành công, {error_cnt} lỗi.")
    
    if error_cnt > 0 and len(error_samples) > 0:
        print("\n⚠️ NGUYÊN NHÂN LỖI (Mẫu 3 dòng đầu tiên):")
        for idx, orig_path, err in error_samples:
            print(f" - Row {idx} | audio_path gốc: '{orig_path}' | Lỗi: {err}")
            
    return success_cnt > 0


# ============================================================
# STEP 3B: FEATURE EXTRACTION
# ============================================================
def calculate_statistics(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values))
    }


def extract_all_features(audio_path):
    y, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    if len(y) == 0:
        raise ValueError("Audio empty")

    features = {"feature_audio_duration": float(len(y) / sr), "feature_sampling_rate": int(sr)}

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(N_MFCC):
        features.update(calculate_statistics(mfcc[i], f"mfcc_{i+1}"))
        features.update(calculate_statistics(delta[i], f"delta_mfcc_{i+1}"))
        features.update(calculate_statistics(delta2[i], f"delta_delta_mfcc_{i+1}"))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean, mel_std = np.mean(mel_db, axis=1), np.std(mel_db, axis=1)
    for i in range(N_MELS):
        features[f"mel_band_{i+1}_mean"] = float(mel_mean[i])
        features[f"mel_band_{i+1}_std"] = float(mel_std[i])

    # Spectral Features
    features.update(calculate_statistics(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH), "spectral_centroid"))
    features.update(calculate_statistics(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH), "spectral_bandwidth"))
    features.update(calculate_statistics(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH), "spectral_rolloff"))
    features.update(calculate_statistics(librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH), "zero_crossing_rate"))
    features.update(calculate_statistics(librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH), "rms_energy"))

    # Pitch / F0
    try:
        f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, frame_length=N_FFT, hop_length=HOP_LENGTH)
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        features.update(calculate_statistics(f0, "pitch_f0"))
    except Exception:
        features.update({"pitch_f0_mean": np.nan, "pitch_f0_std": np.nan, "pitch_f0_min": np.nan, "pitch_f0_max": np.nan})

    return features


def run_step_3b():
    print("\n" + "=" * 70 + "\nSTEP 3B - AUDIO FEATURE EXTRACTION\n" + "=" * 70)
    if not AUDIO_METADATA_PATH.exists():
        print(f"❌ KHÔNG TÌM THẤY {AUDIO_METADATA_PATH}.")
        return

    df = pd.read_csv(AUDIO_METADATA_PATH)
    path_col = "processed_audio_path" if "processed_audio_path" in df.columns else "audio_path"

    feature_records = []
    success_count, error_count = 0, 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        record = row.to_dict()
        status_3a = str(row.get("audio_status", "success")).lower()
        raw_path = str(row.get(path_col, "")).strip()
        
        audio_file = Path(raw_path)
        if not audio_file.is_absolute():
            audio_file = PROJECT_ROOT / audio_file

        if status_3a == "success" and raw_path and audio_file.exists():
            try:
                feats = extract_all_features(audio_file)
                record.update(feats)
                record["feature_status"] = "success"
                record["feature_error"] = ""
                success_count += 1
            except Exception as e:
                record["feature_status"] = "error"
                record["feature_error"] = str(e)
                error_count += 1
        else:
            record["feature_status"] = "error"
            record["feature_error"] = f"File không tồn tại hoặc Step 3A lỗi: {audio_file}"
            error_count += 1

        feature_records.append(record)

    out_df = pd.DataFrame(feature_records)
    out_df.to_csv(AUDIO_FEATURES_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ STEP 3B HOÀN THÀNH: {success_count} thành công, {error_count} lỗi.")


if __name__ == "__main__":
    if run_step_3a():
        run_step_3b()
