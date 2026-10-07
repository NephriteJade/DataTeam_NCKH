import os
import json
import pandas as pd
import config

def main():
    print("=== STEP 8: LƯU TRỮ VÀ KHỞI TẠO MANIFEST DATASET ===")
    
    input_path = config.PROCESSED_DIR / "step7" / "normalized_dataset.csv"
    if not input_path.exists():
        print(f"❌ Không tìm thấy {input_path}")
        return

    df = pd.read_csv(input_path)
    manifest = []

    for idx, row in df.iterrows():
        entry = {
            "id": row.get("segment_id", f"seg_{idx}"),
            "participant_id": row.get("participant_id", ""),
            "text": row.get("text", ""),
            "audio_clip": row.get("processed_audio_path", ""),
            "label_phq_binary": row.get("PHQ_Binary", None),
            "label_phq_score": row.get("PHQ_Score", None)
        }
        manifest.append(entry)

    output_dir = config.PROCESSED_DIR / "step8"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"✅ Hoàn thành Step 8! Manifest JSON tổng hợp lưu tại:")
    print(f"📄 {manifest_path}")

if __name__ == "__main__":
    main()