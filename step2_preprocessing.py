
import os
import pandas as pd
import config

def clean_text(text):
    """Làm sạch văn bản"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    text = " ".join(text.split())  # Loại khoảng trắng thừa
    return text

def detect_text_column(df):
    """Tự động tìm cột chứa nội dung văn bản"""
    possible_cols = ['Text', 'text', 'Transcript', 'transcript', 'Content', 'content', 'Utterance']
    for col in possible_cols:
        if col in df.columns:
            return col
    return df.columns[0]  # Nếu không tìm thấy, lấy cột đầu tiên

def main():
    print("=== BẮT ĐẦU STEP 2: MULTIMODAL PREPROCESSING ===\n")
    
    # Đọc kết quả từ Step 1
    step1_path = os.path.join(config.BASE_DIR, "processed", "step1_participants.csv")
    if not os.path.exists(step1_path):
        print("❌ Chưa có file step1_participants.csv. Hãy chạy Step 1 trước.")
        return
    
    df_step1 = pd.read_csv(step1_path)
    print(f"Đọc được {len(df_step1)} participant từ Step 1.")
    
    metadata_records = []
    
    for _, row in df_step1.iterrows():
        dataset = row['dataset']
        participant = row['participant']
        transcript_path = row['transcript_path']
        
        print(f"Xử lý: {dataset} - {participant}")
        
        # Đọc transcript
        try:
            transcript_df = pd.read_csv(transcript_path)
            text_col = detect_text_column(transcript_df)
            
            for idx, t_row in transcript_df.iterrows():
                text = clean_text(t_row[text_col])
                if not text:
                    continue
                
                segment_id = f"{participant}_{idx}"
                
                metadata_records.append({
                    "dataset": dataset,
                    "participant": participant,
                    "segment_id": segment_id,
                    "text": text,
                    "audio_path": row.get('audio_path', 'Not_available'),
                    "video_path": row.get('video_path', 'Not_available'),
                    "start_time": t_row.get('start', None),
                    "end_time": t_row.get('end', None),
                    "processing_status": "ready_for_feature_extraction"
                })
        except Exception as e:
            print(f"  Lỗi khi đọc transcript của {participant}: {e}")
    
    # Lưu metadata
    if metadata_records:
        metadata_df = pd.DataFrame(metadata_records)
        output_dir = os.path.join(config.BASE_DIR, "processed")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "metadata.csv")
        metadata_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n✅ HOÀN THÀNH STEP 2!")
        print(f"   Tổng segment: {len(metadata_df)}")
        print(f"   File metadata: {output_path}")
    else:
        print("⚠️ Không tạo được metadata.")

if __name__ == "__main__":
    main()
