import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config

def main():
    print("=== STEP 7: CHUẨN HÓA ĐẶC TRƯNG (SCALING) ===")
    
    input_path = config.PROCESSED_DIR / "step6" / "synchronized_dataset.csv"
    if not input_path.exists():
        print(f"❌ Không tìm thấy {input_path}")
        return

    df = pd.read_csv(input_path)
    
    exclude_cols = [
        'participant', 'participant_id', 'segment_id', 'text', 'audio_path', 
        'speaker', 'processed_audio_path', 'audio_status', 'sync_status',
        'PHQ_Binary', 'PHQ_Score', 'Gender'
    ]
    
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0))

    output_dir = config.PROCESSED_DIR / "step7"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "normalized_dataset.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ Hoàn thành Step 7. Đã chuẩn hóa {len(feature_cols)} đặc trưng.")
    print(f"📄 Lưu tại: {output_path}")

if __name__ == "__main__":
    main()
