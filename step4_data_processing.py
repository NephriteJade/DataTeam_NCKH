import os
import numpy as np
import pandas as pd
import config

def main():
    print("=== STEP 4: LÀM SẠCH VÀ CHUẨN HÓA DỮ LIỆU ĐẶC TRƯNG ===")
    
    input_path = config.PROCESSED_DIR / "step3" / "audio_features.csv"
    if not input_path.exists():
        print(f"❌ Không tìm thấy {input_path}")
        return

    df = pd.read_csv(input_path)
    
    # Chỉ làm sạch trên các dòng trích xuất audio thành công
    df = df[df['audio_status'] == 'success'].copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Loại các cột timestamp gốc khỏi danh sách feature số
    exclude_cols = ['start_time', 'end_time']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Thay thế Inf -> NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Impute Median cho các giá trị thiếu
    for col in feature_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Bỏ các cột có variance = 0
    zero_var_cols = [col for col in feature_cols if df[col].var() == 0 or pd.isna(df[col].var())]
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)

    output_dir = config.PROCESSED_DIR / "step4"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "audio_features_final.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ Hoàn thành Step 4. Tập dữ liệu sạch có {len(df)} dòng, {len(df.columns)} cột.")
    print(f"📄 Đã lưu tại: {output_path}")

if __name__ == "__main__":
    main()
