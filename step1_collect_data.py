
import os
import pandas as pd
from pathlib import Path
import config

def check_file_exists(directory, patterns):
    """Kiểm tra sự tồn tại của file theo pattern"""
    for pattern in patterns:
        for file in os.listdir(directory):
            if file.lower().endswith(pattern.lower()) or pattern.lower() in file.lower():
                return os.path.join(directory, file)
    return None

def main():
    print("=== BẮT ĐẦU STEP 1: THU THẬP VÀ KIỂM TRA DỮ LIỆU ===\n")
    
    all_participants = []
    
    for dataset_name, dataset_info in config.DATASETS.items():
        dataset_path = dataset_info["path"]
        print(f"--- Checking dataset: {dataset_name} ---")
        
        if not os.path.exists(dataset_path):
            print(f"  → Dataset {dataset_name} chưa tồn tại, bỏ qua.\n")
            continue
            
        # Quét các participant
        participants = [p for p in os.listdir(dataset_path) 
                       if os.path.isdir(os.path.join(dataset_path, p))]
        
        print(f"  Tìm thấy {len(participants)} participant.")
        
        valid_count = 0
        for participant in sorted(participants):
            part_path = os.path.join(dataset_path, participant)
            
            # Kiểm tra file theo pattern DAIC
            transcript_path = check_file_exists(part_path, ["_Transcript.csv", "transcript.csv"])
            audio_path = check_file_exists(part_path, ["_AUDIO.wav", ".wav"])
            video_path = check_file_exists(part_path, [".mp4", ".avi"])
            
            if transcript_path:
                valid_count += 1
                all_participants.append({
                    "dataset": dataset_name,
                    "participant": participant,
                    "transcript_path": transcript_path,
                    "audio_path": audio_path or "Not_available",
                    "video_path": video_path or "Not_available",
                    "status": "valid"
                })
                print(f"  ✓ Valid: {participant}")
            else:
                print(f"  → Participant {participant} thiếu transcript")
        
        print(f"  → Valid participants: {valid_count}\n")
    
    # Lưu kết quả
    if all_participants:
        df = pd.DataFrame(all_participants)
        output_dir = os.path.join(config.BASE_DIR, "processed")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "step1_participants.csv")
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"✅ HOÀN THÀNH STEP 1!")
        print(f"   Tổng valid participants: {len(df)}")
        print(f"   File kết quả: {output_path}")
    else:
        print("⚠️ Không tìm thấy participant hợp lệ nào.")

if __name__ == "__main__":
    main()