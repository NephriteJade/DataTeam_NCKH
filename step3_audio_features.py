import os
import warnings
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import config

warnings.filterwarnings("ignore")

STEP3_DIR = config.PROCESSED_DIR / "step3"
AUDIO_SEGMENTS_DIR = STEP3_DIR / "audio_segments"
STEP3_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000

def extract_features(y, sr):
    feats = {"duration": len(y) / sr}
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        
    # Spectral Centroid & ZCR
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats["spectral_centroid_mean"] = float(np.mean(sc))
    feats["spectral_centroid_std"] = float(np.std(sc))
    
    zcr = librosa.feature.zero_crossing_rate(y)
    feats["zcr_mean"] = float(np.mean(zcr))
    
    # Energy RMS
    rms = librosa.feature.rms(y=y)
    feats["rms_mean"] = float(np.mean(rms))
    
    return feats

def main():
    print("=== STEP 3: TÁCH ĐOẠN ÂM THANH & TRÍCH XUẤT ĐẶC TRƯNG ===")
    
    metadata_path = config.PROCESSED_DIR / "metadata.csv"
    if not metadata_path.exists():
        print("❌ Không tìm thấy metadata.csv. Chạy Step 2 trước!")
        return

    df = pd.read_csv(metadata_path)
    feature_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extrating Audio Features"):
        audio_path = row['audio_path']
        start = row['start_time']
        end = row['end_time']
        seg_id = row['segment_id']

        rec = row.to_dict()
        
        try:
            if pd.notna(start) and pd.notna(end) and float(end) > float(start):
                start, end = float(start), float(end)
                duration = end - start
                
                # Load đoạn audio theo timestamp
                y, sr = librosa.load(audio_path, sr=TARGET_SR, offset=start, duration=duration)
                
                if len(y) > 0:
                    # Lưu đoạn wav
                    seg_wav_path = AUDIO_SEGMENTS_DIR / f"{seg_id}.wav"
                    sf.write(str(seg_wav_path), y, sr)
                    
                    # Trích xuất đặc trưng
                    feats = extract_features(y, sr)
                    rec.update(feats)
                    rec["processed_audio_path"] = str(seg_wav_path)
                    rec["audio_status"] = "success"
                else:
                    rec["audio_status"] = "empty"
            else:
                rec["audio_status"] = "invalid_timestamp"
        except Exception as e:
            rec["audio_status"] = "error"
            rec["audio_error"] = str(e)
            
        feature_records.append(rec)

    out_df = pd.DataFrame(feature_records)
    out_path = STEP3_DIR / "audio_features.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ Hoàn thành Step 3. File lưu tại: {out_path}")

if __name__ == "__main__":
    main()
