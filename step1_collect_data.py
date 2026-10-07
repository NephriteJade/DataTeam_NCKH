import os
import pandas as pd
import config

def main():
    print("=== STEP 1: THU THẬP VÀ KIỂM TRA DỮ LIỆU E-DAIC ===")
    
    dataset_path = config.DATASET_DIR
    if not dataset_path.exists():
        print(f"❌ Thư mục {dataset_path} không tồn tại!")
        return

    participants = [p for p in os.listdir(dataset_path) if os.path.isdir(dataset_path / p)]
    print(f"Tìm thấy {len(participants)} thư mục participant.")

    records = []
    for p in sorted(participants):
        p_path = dataset_path / p
        p_id = p.replace("_P", "")
        
        audio_file = p_path / f"{p_id}_AUDIO.wav"
        transcript_file = p_path / f"{p_id}_Transcript.csv"
        features_dir = p_path / "features"
        
        has_audio = audio_file.exists()
        has_transcript = transcript_file.exists()
        
        if has_transcript and has_audio:
            records.append({
                "participant": p,
                "participant_id": p_id,
                "audio_path": str(audio_file),
                "transcript_path": str(transcript_file),
                "has_features": features_dir.exists(),
                "status": "valid"
            })

    df = pd.DataFrame(records)
    output_path = config.PROCESSED_DIR / "step1_participants.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ Hoàn thành Step 1. Số participant hợp lệ: {len(df)}")
    print(f"📄 Kết quả lưu tại: {output_path}")

if __name__ == "__main__":
    main()
