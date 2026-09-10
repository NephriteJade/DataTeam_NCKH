import os
import pandas as pd
import numpy as np

BASE_DIR = "/Applications/NCKH_DATATEAM"

INPUT_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "step4",
    "audio_features_final.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed",
    "step5"
)

STATISTICS_PATH = os.path.join(
    OUTPUT_DIR,
    "step5c_feature_statistics.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "step5c_statistics_summary.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 5C - STATISTICAL ANALYSIS")
    print("=" * 70)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy dataset:\n{INPUT_PATH}"
        )

    print("\nĐang đọc dữ liệu...")
    print(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nTổng số dòng: {len(df)}")
    print(f"Tổng số cột: {len(df.columns)}")

    exclude_columns = [
        "participant",
        "segment_id",
        "dataset",
        "text",
        "audio_path",
        "video_path",
        "processing_status",
        "processed_audio_path",
        "resolved_original_audio_path",
        "audio_status",
        "feature_status",
        "audio_sampling_rate",
        "feature_sampling_rate"
    ]

    feature_columns = [
        column
        for column in df.select_dtypes(
            include=[np.number]
        ).columns
        if column not in exclude_columns
    ]

    print(f"\nTổng audio features: {len(feature_columns)}")

    statistics = []

    for column in feature_columns:
        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        statistics.append({
            "feature": column,
            "count": int(series.count()),
            "missing_count": int(series.isna().sum()),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "q25": series.quantile(0.25),
            "median": series.median(),
            "q75": series.quantile(0.75),
            "max": series.max(),
            "variance": series.var(),
            "unique_count": int(series.nunique()),
            "zero_variance": bool(
                series.var() == 0
            )
        })

    statistics_df = pd.DataFrame(statistics)

    statistics_df = statistics_df.round(6)

    statistics_df.to_csv(
        STATISTICS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_audio_features": len(feature_columns),
        "features_with_missing": int(
            (
                statistics_df["missing_count"] > 0
            ).sum()
        ),
        "features_with_zero_variance": int(
            statistics_df["zero_variance"].sum()
        ),
        "features_with_single_value": int(
            (
                statistics_df["unique_count"] <= 1
            ).sum()
        ),
        "total_missing_values": int(
            statistics_df["missing_count"].sum()
        )
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("STEP 5C COMPLETED")
    print("=" * 70)

    print(f"\nTổng dòng: {len(df)}")
    print(f"Tổng cột: {len(df.columns)}")
    print(f"Tổng audio features: {len(feature_columns)}")

    print(
        f"Feature có missing: "
        f"{summary['features_with_missing']}"
    )

    print(
        f"Feature variance = 0: "
        f"{summary['features_with_zero_variance']}"
    )

    print(
        f"Tổng missing values trong features: "
        f"{summary['total_missing_values']}"
    )

    print("\nFile thống kê chi tiết:")
    print(STATISTICS_PATH)

    print("\nFile tổng hợp:")
    print(SUMMARY_PATH)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()