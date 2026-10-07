import os
import pandas as pd
import config

def main():
    print("=== STEP 5: GÁN NHÃN TRẦM CẢM (PHQ-8) ===")
    
    dataset_path = config.PROCESSED_DIR / "step4" / "audio_features_final.csv"
    train_label_path = config.LABELS_DIR / "train_split.csv"
    dev_label_path = config.LABELS_DIR / "dev_split.csv"

    if not dataset_path.exists():
        print(f"❌ Không tìm thấy {dataset_path}")
        return

    df = pd.read_csv(dataset_path)

    # Tải nhãn nếu tồn tại
    labels_list = []
    for path in [train_label_path, dev_label_path]:
        if path.exists():
            lbl_df = pd.read_csv(path)
            labels_list.append(lbl_df)
            
    if labels_list:
        all_labels = pd.concat(labels_list, ignore_index=True)
        all_labels.columns = [c.strip() for c in all_labels.columns]
        
        # Chuẩn hóa Participant_ID để khớp khóa chính
        all_labels['participant_id'] = all_labels['Participant_ID'].astype(str).str.replace('_P', '')
        df['participant_id'] = df['participant_id'].astype(str)

        merged_df = df.merge(
            all_labels[['participant_id', 'PHQ_Binary', 'PHQ_Score', 'Gender']], 
            on='participant_id', 
            how='left'
        )
    else:
        print("⚠️ Chưa tìm thấy file nhãn trong thư mục /labels. Giữ nguyên dataset.")
        merged_df = df

    output_dir = config.PROCESSED_DIR / "step5"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "labeled_dataset.csv"
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ Hoàn thành Step 5. Đã lưu tập dữ liệu gán nhãn tại: {output_path}")

if __name__ == "__main__":
    main()
