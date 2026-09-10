# config.py
import os

BASE_DIR = r"D:\NCKH_DATATEAM\676_P\676_P"   # ← Sửa thành dòng này

print(f"DEBUG - BASE_DIR = {BASE_DIR}")

DATASETS = {
    "E-DAIC": {
        "path": os.path.join(BASE_DIR, "dataset", "E-DAIC"),
        "description": "Extended DAIC-WOZ"
    },
    "Vietnam": {
        "path": os.path.join(BASE_DIR, "dataset", "Vietnam"),
        "description": "Dữ liệu Việt Nam"
    },
    "Japan": {
        "path": os.path.join(BASE_DIR, "dataset", "Japan"),
        "description": "Dữ liệu Nhật Bản"
    }
}

# Debug
if __name__ == "__main__":
    for name, info in DATASETS.items():
        print(f"DEBUG {name} path: {info['path']}")
        print(f"DEBUG Exists? {os.path.exists(info['path'])}")