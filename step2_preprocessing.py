import os
import pandas as pd
import config

def read_transcript(file_path):
    # E-DAIC transcript thường dùng tab '\t' làm phân cách
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8-sig')
        if len(df.columns) <= 1:
            df = pd.read_csv(file_path, sep=',', encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(file_path, sep=r'\s+', engine='python')
    return df

def main():
    print("=== STEP 2: TIỀN XỬ LÝ TRANSCRIPT VÀ TẠO SEGMENTS ===")
    
    step1_path = config.PROCESSED_DIR / "step1_participants.csv"
    if not step1_path.exists():
        print("❌ Chưa có file step1_participants.csv. Chạy Step 1 trước!")
        return

    df_step1 = pd.read_csv(step1_path)
    metadata_records = []

    for _, row in df_step1.iterrows():
        p_id = row['participant_id']
        t_path = row['transcript_path']
        a_path = row['audio_path']

        try:
            t_df = read_transcript(t_path)
            
            # Chuẩn hóa tên cột
            t_df.columns = [c.strip().lower() for c in t_df.columns]
            
            text_col = 'value' if 'value' in t_df.columns else ('text' if 'text' in t_df.columns else t_df.columns[-1])
            start_col = 'start_time' if 'start_time' in t_df.columns else 'start'
            end_col = 'end_time' if 'end_time' in t_df.columns else 'end'

            for idx, r in t_df.iterrows():
                text = str(r.get(text_col, '')).strip()
                if not text or pd.isna(text):
                    continue

                segment_id = f"{p_id}_{idx}"
                metadata_records.append({
                    "participant": row['participant'],
                    "participant_id": p_id,
                    "segment_id": segment_id,
                    "text": text,
                    "start_time": r.get(start_col, None),
                    "end_time": r.get(end_col, None),
                    "audio_path": a_path,
                    "speaker": r.get('speaker', 'Ellie' if idx % 2 == 0 else 'Participant')
                })
        except Exception as e:
            print(f"⚠️ Lỗi đọc transcript {p_id}: {e}")

    metadata_df = pd.DataFrame(metadata_records)
    output_path = config.PROCESSED_DIR / "metadata.csv"
    metadata_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ Hoàn thành Step 2. Tổng số segments: {len(metadata_df)}")
    print(f"📄 File lưu tại: {output_path}")

if __name__ == "__main__":
    main()
