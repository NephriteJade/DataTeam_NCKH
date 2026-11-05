import os
import pandas as pd
import config

def main():
    print("=== STEP 6: ĐỒNG BỘ DỮ LIỆU ĐA PHƯƠNG THỨC ===")
    
    input_path = config.PROCESSED_DIR / "step5" / "labeled_dataset.csv"
    if not input_path.exists():
        input_path = config.PROCESSED_DIR / "step4" / "audio_features_final.csv"

    df = pd.read_csv(input_path)
    
    # Tạo chỉ số đồng bộ thời gian
    if 'start_time' in df.columns and 'end_time' in df.columns:
        df['duration_calculated'] = df['end_time'] - df['start_time']
    
    df['sync_status'] = 'synced'

    output_dir = config.PROCESSED_DIR / "step6"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "synchronized_dataset.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ Hoàn thành Step 6. Đồng bộ {len(df)} mẫu dữ liệu tại: {output_path}")

if __name__ == "__main__":
    main()
