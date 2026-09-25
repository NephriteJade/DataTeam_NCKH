import os
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "processed",
    "step3",
    "audio_features.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "processed",
    "step4"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "audio_features_cleaned.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step4_quality_report.csv"
)

MISSING_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step4_missing_report.csv"
)

REMOVED_COLUMNS_PATH = os.path.join(
    OUTPUT_DIR,
    "step4_removed_columns.csv"
)

ID_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step4_id_quality_report.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# METADATA KEYWORDS
# ============================================================

METADATA_KEYWORDS = [
    "participant",
    "subject",
    "segment",
    "audio",
    "path",
    "file",
    "label",
    "depression",
    "phq",
    "split",
    "gender",
    "age",
    "duration",
    "sampling_rate",
    "status",
    "error",
    "text",
    "timestamp",
    "start_time",
    "end_time"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 4 - DATA QUALITY CHECK & FEATURE PROCESSING")
    print("=" * 70)

    # ========================================================
    # 1. CHECK INPUT
    # ========================================================

    if not os.path.exists(INPUT_PATH):

        print()
        print("❌ Không tìm thấy file:")
        print(INPUT_PATH)
        print()
        print("Hãy kiểm tra Step 3B trước.")
        return

    print()
    print("Đang đọc dữ liệu...")
    print(INPUT_PATH)

    df = pd.read_csv(
        INPUT_PATH
    )

    original_rows = len(df)
    original_columns = len(df.columns)

    print()
    print(
        f"Số dòng ban đầu: {original_rows}"
    )

    print(
        f"Số cột ban đầu: {original_columns}"
    )


    # ========================================================
    # 2. IDENTIFY NUMERIC / NON-NUMERIC COLUMNS
    # ========================================================

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    non_numeric_columns = df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()


    # ========================================================
    # 3. IDENTIFY METADATA COLUMNS
    # ========================================================

    metadata_columns = []

    for column in df.columns:

        column_lower = str(
            column
        ).lower()

        if any(
            keyword in column_lower
            for keyword in METADATA_KEYWORDS
        ):

            metadata_columns.append(
                column
            )


    # ========================================================
    # 4. IDENTIFY FEATURE COLUMNS
    # ========================================================

    feature_columns = [
        column
        for column in numeric_columns
        if column not in metadata_columns
    ]


    print()
    print(
        f"Numeric columns : {len(numeric_columns)}"
    )

    print(
        f"Metadata columns: {len(metadata_columns)}"
    )

    print(
        f"Feature columns : {len(feature_columns)}"
    )


    # ========================================================
    # 5. MISSING VALUE REPORT - BEFORE
    # ========================================================

    missing_report = []

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        missing_percentage = (
            missing_count
            / len(df)
            * 100
        )

        missing_report.append({

            "column": column,

            "missing_count":
                missing_count,

            "missing_percentage":
                missing_percentage,

            "data_type":
                str(df[column].dtype),

            "column_type":
                (
                    "feature"
                    if column in feature_columns
                    else (
                        "metadata"
                        if column in metadata_columns
                        else "other"
                    )
                )

        })

    missing_df = pd.DataFrame(
        missing_report
    )

    missing_df.to_csv(
        MISSING_REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 6. CHECK INF VALUES
    # ========================================================

    inf_report = []

    for column in numeric_columns:

        values = df[column].to_numpy()

        inf_count = int(
            np.isinf(values).sum()
        )

        inf_report.append({

            "column": column,

            "inf_count":
                inf_count

        })

    inf_df = pd.DataFrame(
        inf_report
    )

    inf_columns = inf_df[
        inf_df["inf_count"] > 0
    ]["column"].tolist()


    # ========================================================
    # 7. REPLACE INF -> NaN
    # ========================================================

    if len(inf_columns) > 0:

        print()
        print(
            f"⚠️ Phát hiện {len(inf_columns)} "
            f"cột chứa Inf."
        )

        for column in inf_columns:

            df[column] = df[column].replace(
                [np.inf, -np.inf],
                np.nan
            )

    else:

        print()
        print(
            "✓ Không phát hiện Inf values."
        )


    # ========================================================
    # 8. REMOVE ALL-MISSING COLUMNS
    # ========================================================

    all_missing_columns = []

    for column in df.columns:

        if df[column].isna().all():

            all_missing_columns.append(
                column
            )


    print()

    if len(all_missing_columns) > 0:

        print(
            f"⚠️ Phát hiện "
            f"{len(all_missing_columns)} "
            f"cột bị Missing 100%."
        )

        print()
        print(
            "Các cột sẽ bị loại bỏ:"
        )

        for column in all_missing_columns:

            print(
                f"   - {column}"
            )

        df = df.drop(
            columns=all_missing_columns
        )

    else:

        print(
            "✓ Không có cột Missing 100%."
        )


    # ========================================================
    # 9. UPDATE FEATURE COLUMNS AFTER REMOVAL
    # ========================================================

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    metadata_columns = [
        column
        for column in df.columns
        if any(
            keyword in str(column).lower()
            for keyword in METADATA_KEYWORDS
        )
    ]

    feature_columns = [
        column
        for column in numeric_columns
        if column not in metadata_columns
    ]


    # ========================================================
    # 10. REMOVE ZERO-VARIANCE FEATURES
    # ========================================================

    zero_variance_columns = []

    for column in feature_columns:

        variance = df[column].var(
            skipna=True
        )

        if (
            pd.isna(variance)
            or variance == 0
        ):

            zero_variance_columns.append(
                column
            )


    if len(zero_variance_columns) > 0:

        print()
        print(
            f"⚠️ Loại bỏ "
            f"{len(zero_variance_columns)} "
            f"feature có variance = 0."
        )

        for column in zero_variance_columns:

            print(
                f"   - {column}"
            )

        df = df.drop(
            columns=zero_variance_columns
        )

    else:

        print()
        print(
            "✓ Không có feature variance = 0."
        )


    # ========================================================
    # 11. UPDATE FEATURE COLUMNS AGAIN
    # ========================================================

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    metadata_columns = [
        column
        for column in df.columns
        if any(
            keyword in str(column).lower()
            for keyword in METADATA_KEYWORDS
        )
    ]

    feature_columns = [
        column
        for column in numeric_columns
        if column not in metadata_columns
    ]


    # ========================================================
    # 12. MEDIAN IMPUTATION
    # ========================================================

    imputation_report = []

    for column in feature_columns:

        missing_before = int(
            df[column].isna().sum()
        )

        if missing_before == 0:
            continue

        median_value = df[column].median()

        # Trường hợp đặc biệt:
        # nếu median vẫn NaN thì bỏ cột
        if pd.isna(median_value):

            print()
            print(
                f"⚠️ Không thể tính median cho: "
                f"{column}"
            )

            continue

        df[column] = df[column].fillna(
            median_value
        )

        imputation_report.append({

            "column":
                column,

            "missing_before":
                missing_before,

            "median_used":
                float(median_value),

            "missing_after":
                int(
                    df[column].isna().sum()
                )

        })


    # ========================================================
    # 13. HANDLE REMAINING ALL-MISSING COLUMNS
    # ========================================================

    remaining_all_missing = []

    for column in df.columns:

        if df[column].isna().all():

            remaining_all_missing.append(
                column
            )


    if len(remaining_all_missing) > 0:

        print()
        print(
            "⚠️ Vẫn còn cột Missing 100%:"
        )

        for column in remaining_all_missing:

            print(
                f"   - {column}"
            )

        df = df.drop(
            columns=remaining_all_missing
        )


    # ========================================================
    # 14. FINAL MISSING CHECK
    # ========================================================

    final_missing_count = int(
        df.isna().sum().sum()
    )


    # ========================================================
    # 15. FINAL INF CHECK
    # ========================================================

    final_inf_count = 0

    final_numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
    )

    for column in final_numeric_columns:

        final_inf_count += int(
            np.isinf(
                df[column].to_numpy()
            ).sum()
        )


    # ========================================================
    # 16. DUPLICATE ROWS
    # ========================================================

    duplicate_rows = int(
        df.duplicated().sum()
    )


    # ========================================================
    # 17. DUPLICATE ID CHECK
    # ========================================================

    possible_id_columns = [

        column

        for column in df.columns

        if (

            "participant_id"
            in column.lower()

            or

            "subject_id"
            in column.lower()

            or

            "segment_id"
            in column.lower()

            or

            column.lower()
            == "participant"

            or

            column.lower()
            == "subject"

        )

    ]

    duplicate_id_data = []

    for column in possible_id_columns:

        duplicate_id_data.append({

            "column":
                column,

            "unique_values":
                df[column].nunique(),

            "total_rows":
                len(df),

            "duplicate_values":
                int(
                    df[column].duplicated().sum()
                )

        })

    duplicate_id_report = pd.DataFrame(
        duplicate_id_data
    )


    # ========================================================
    # 18. REMOVED COLUMNS REPORT
    # ========================================================

    removed_columns = []

    for column in all_missing_columns:

        removed_columns.append({

            "column":
                column,

            "reason":
                "100% missing"

        })

    for column in zero_variance_columns:

        removed_columns.append({

            "column":
                column,

            "reason":
                "zero variance"

        })

    for column in remaining_all_missing:

        if column not in all_missing_columns:

            removed_columns.append({

                "column":
                    column,

                "reason":
                    "100% missing after processing"

            })


    removed_columns_df = pd.DataFrame(
        removed_columns
    )

    removed_columns_df.to_csv(
        REMOVED_COLUMNS_PATH,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 19. QUALITY REPORT
    # ========================================================

    quality_report = pd.DataFrame({

        "metric": [

            "original_rows",

            "original_columns",

            "final_rows",

            "final_columns",

            "numeric_columns",

            "metadata_columns",

            "feature_columns",

            "inf_columns_before_cleaning",

            "all_missing_columns_removed",

            "zero_variance_columns_removed",

            "duplicate_rows",

            "final_missing_values",

            "final_inf_values"

        ],

        "value": [

            original_rows,

            original_columns,

            len(df),

            len(df.columns),

            len(numeric_columns),

            len(metadata_columns),

            len(feature_columns),

            len(inf_columns),

            len(all_missing_columns)
            + len(remaining_all_missing),

            len(zero_variance_columns),

            duplicate_rows,

            final_missing_count,

            final_inf_count

        ]

    })


    # ========================================================
    # 20. SAVE DATA
    # ========================================================

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    quality_report.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )


    if len(duplicate_id_report) > 0:

        duplicate_id_report.to_csv(
            ID_REPORT_PATH,
            index=False,
            encoding="utf-8-sig"
        )


    # ========================================================
    # 21. FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 4 COMPLETED")
    print("=" * 70)

    print()

    print(
        f"Số dòng ban đầu       : "
        f"{original_rows}"
    )

    print(
        f"Số cột ban đầu        : "
        f"{original_columns}"
    )

    print(
        f"Số dòng sau xử lý     : "
        f"{len(df)}"
    )

    print(
        f"Số cột sau xử lý      : "
        f"{len(df.columns)}"
    )

    print(
        f"Inf columns            : "
        f"{len(inf_columns)}"
    )

    print(
        f"All-missing columns    : "
        f"{len(all_missing_columns) + len(remaining_all_missing)}"
    )

    print(
        f"Zero-variance columns  : "
        f"{len(zero_variance_columns)}"
    )

    print(
        f"Duplicate rows         : "
        f"{duplicate_rows}"
    )

    print(
        f"Missing còn lại        : "
        f"{final_missing_count}"
    )

    print(
        f"Inf còn lại            : "
        f"{final_inf_count}"
    )

    print()

    print(
        "File dữ liệu:"
    )

    print(
        OUTPUT_PATH
    )

    print()

    print(
        "Quality report:"
    )

    print(
        REPORT_PATH
    )

    print()

    print(
        "Missing report:"
    )

    print(
        MISSING_REPORT_PATH
    )

    print()

    print(
        "Removed columns report:"
    )

    print(
        REMOVED_COLUMNS_PATH
    )

    if len(duplicate_id_report) > 0:

        print()

        print(
            "ID quality report:"
        )

        print(
            ID_REPORT_PATH
        )

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
