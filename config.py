import os
from pathlib import Path

# Đường dẫn gốc dự án
BASE_DIR = Path(r"D:\NCKH_DATATEAM").resolve()

# Cấu hình các đường dẫn con
DATASET_DIR = BASE_DIR / "dataset" / "E-DAIC"
LABELS_DIR = BASE_DIR / "labels"
PROCESSED_DIR = BASE_DIR / "processed"

# Đảm bảo các thư mục đầu ra tồn tại
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATASET_DIR: {DATASET_DIR} (Exists: {DATASET_DIR.exists()})")
