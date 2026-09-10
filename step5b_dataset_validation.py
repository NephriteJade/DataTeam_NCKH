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

VALIDATION_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step5b_validation_report.csv"
)

FEATURE_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step5b_feature_quality_report.csv"
)

PARTICIPANT_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step5b_participant_quality_report.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 5B - DATASET VALIDATION")
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

    validation = []

    validation.append({
        "check": "total_rows",
        "value": len(df),
        "status": "PASS"
    })

    validation.append({
        "check": "total_columns",
        "value": len(df.columns),
        "status": "PASS"
    })

    if "participant" in df.columns:
        missing_participant = int(df["participant"].isna().sum())
        duplicate_participant = int(df["participant"].duplicated().sum())
        participant_count = int(df["participant"].nunique())

        validation.append({
            "check": "participant_count",
            "value": participant_count,
            "status": "PASS" if participant_count > 0 else "FAIL"
        })

        validation.append({
            "check": "missing_participant",
            "value": missing_participant,
            "status": "PASS" if missing_participant == 0 else "FAIL"
        })

        validation.append({
            "check": "duplicate_participant_rows",
            "value": duplicate_participant,
            "status": "INFO"
        })

    if "segment_id" in df.columns:
        missing_segment = int(df["segment_id"].isna().sum())
        duplicate_segment = int(df["segment_id"].duplicated().sum())
        unique_segment = int(df["segment_id"].nunique())

        validation.append({
            "check": "unique_segment_id",
            "value": unique_segment,
            "status": "PASS" if unique_segment == len(df) else "FAIL"
        })

        validation.append({
            "check": "missing_segment_id",
            "value": missing_segment,
            "status": "PASS" if missing_segment == 0 else "FAIL"
        })

        validation.append({
            "check": "duplicate_segment_id",
            "value": duplicate_segment,
            "status": "PASS" if duplicate_segment == 0 else "FAIL"
        })

    missing_total = int(df.isna().sum().sum())

    validation.append({
        "check": "missing_values_total",
        "value": missing_total,
        "status": "PASS" if missing_total == 0 else "FAIL"
    })

    inf_total = 0

    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) > 0:
        inf_total = int(
            np.isinf(numeric_df.to_numpy()).sum()
        )

    validation.append({
        "check": "inf_values_total",
        "value": inf_total,
        "status": "PASS" if inf_total == 0 else "FAIL"
    })

    duplicate_rows = int(df.duplicated().sum())

    validation.append({
        "check": "duplicate_rows",
        "value": duplicate_rows,
        "status": "PASS" if duplicate_rows == 0 else "FAIL"
    })

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    zero_variance_columns = []

    for column in numeric_columns:
        variance = df[column].var()

        if pd.notna(variance) and variance == 0:
            zero_variance_columns.append(column)

    validation.append({
        "check": "zero_variance_columns",
        "value": len(zero_variance_columns),
        "status": "PASS" if len(zero_variance_columns) == 0 else "INFO"
    })

    if "audio_duration" in df.columns:
        invalid_duration = int(
            (
                df["audio_duration"].notna()
                & (df["audio_duration"] <= 0)
            ).sum()
        )

        validation.append({
            "check": "invalid_audio_duration",
            "value": invalid_duration,
            "status": "PASS" if invalid_duration == 0 else "FAIL"
        })

    if "audio_sampling_rate" in df.columns:
        sampling_rates = df["audio_sampling_rate"].dropna().unique()

        validation.append({
            "check": "sampling_rate_types",
            "value": len(sampling_rates),
            "status": "PASS"
        })

        validation.append({
            "check": "sampling_rates",
            "value": ", ".join(
                str(x) for x in sampling_rates
            ),
            "status": "INFO"
        })

    if "feature_status" in df.columns:
        feature_status_missing = int(
            df["feature_status"].isna().sum()
        )

        feature_status_error = int(
            (
                df["feature_status"]
                .astype(str)
                .str.lower()
                .str.contains("error")
            ).sum()
        )

        validation.append({
            "check": "feature_status_missing",
            "value": feature_status_missing,
            "status": "PASS" if feature_status_missing == 0 else "FAIL"
        })

        validation.append({
            "check": "feature_status_error",
            "value": feature_status_error,
            "status": "PASS" if feature_status_error == 0 else "FAIL"
        })

    if "audio_status" in df.columns:
        audio_status_missing = int(
            df["audio_status"].isna().sum()
        )

        validation.append({
            "check": "audio_status_missing",
            "value": audio_status_missing,
            "status": "PASS" if audio_status_missing == 0 else "FAIL"
        })

    validation_df = pd.DataFrame(validation)

    validation_df.to_csv(
        VALIDATION_REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    feature_records = []

    for column in numeric_columns:
        missing = int(df[column].isna().sum())
        inf_count = int(
            np.isinf(
                df[column].to_numpy()
            ).sum()
        )

        variance = df[column].var()

        feature_records.append({
            "feature": column,
            "dtype": str(df[column].dtype),
            "missing_count": missing,
            "missing_percentage": round(
                missing / len(df) * 100,
                4
            ),
            "inf_count": inf_count,
            "unique_count": int(
                df[column].nunique()
            ),
            "variance": variance,
            "zero_variance": bool(
                pd.notna(variance) and variance == 0
            ),
            "mean": df[column].mean(),
            "std": df[column].std(),
            "min": df[column].min(),
            "max": df[column].max()
        })

    feature_df = pd.DataFrame(feature_records)

    feature_df.to_csv(
        FEATURE_REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    if "participant" in df.columns:
        participant_records = []

        for participant, group in df.groupby(
            "participant",
            dropna=False
        ):
            participant_records.append({
                "participant": participant,
                "segment_count": len(group),
                "unique_segment_count": (
                    group["segment_id"].nunique()
                    if "segment_id" in group.columns
                    else None
                ),
                "missing_values": int(
                    group.isna().sum().sum()
                ),
                "duplicate_rows": int(
                    group.duplicated().sum()
                )
            })

        participant_df = pd.DataFrame(
            participant_records
        )

        participant_df.to_csv(
            PARTICIPANT_REPORT_PATH,
            index=False,
            encoding="utf-8-sig"
        )

    print("\n" + "=" * 70)
    print("STEP 5B COMPLETED")
    print("=" * 70)

    print(f"\nTổng dòng: {len(df)}")
    print(f"Tổng cột: {len(df.columns)}")
    print(f"Tổng participant: {df['participant'].nunique() if 'participant' in df.columns else 'N/A'}")
    print(f"Tổng segment: {len(df)}")
    print(f"Missing values: {missing_total}")
    print(f"Inf values: {inf_total}")
    print(f"Dòng trùng lặp: {duplicate_rows}")
    print(f"Cột numeric: {len(numeric_columns)}")
    print(f"Cột variance = 0: {len(zero_variance_columns)}")

    print("\nCác file báo cáo:")
    print(VALIDATION_REPORT_PATH)
    print(FEATURE_REPORT_PATH)
    print(PARTICIPANT_REPORT_PATH)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()