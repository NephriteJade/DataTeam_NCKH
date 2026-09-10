import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "processed",
    "step4",
    "audio_features_cleaned.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "processed",
    "step4"
)

MISSING_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step4_missing_value_report.csv"
)

MISSING_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "step4_missing_summary_by_group.csv"
)

MISSING_ONLY_PATH = os.path.join(
    OUTPUT_DIR,
    "step4_missing_columns_only.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def classify_column(column):

    column_lower = column.lower()

    metadata_keywords = [

        "participant_id",
        "participant",
        "subject_id",
        "subject",
        "segment_id",
        "segment",
        "audio_path",
        "processed_audio_path",
        "audio",
        "file",
        "path",
        "label",
        "depression",
        "phq",
        "split",
        "gender",
        "sex",
        "age",
        "status",
        "error"

    ]

    feature_keywords = [

        "mfcc",
        "delta_mfcc",
        "delta_delta_mfcc",
        "mel_",
        "spectral_",
        "zero_crossing",
        "rms_",
        "chroma_",
        "pitch_",
        "f0"

    ]

    for keyword in feature_keywords:

        if keyword in column_lower:

            return "audio_feature"

    for keyword in metadata_keywords:

        if keyword in column_lower:

            return "metadata"

    return "other"


def main():

    print()
    print("=" * 70)
    print("STEP 4A - MISSING VALUE ANALYSIS")
    print("=" * 70)

    if not os.path.exists(INPUT_PATH):

        print()
        print("❌ Không tìm thấy file:")
        print(INPUT_PATH)
        return

    print()
    print("Đang đọc dữ liệu...")
    print(INPUT_PATH)

    df = pd.read_csv(
        INPUT_PATH
    )

    total_rows = len(df)

    total_columns = len(df.columns)

    print()
    print(
        f"Tổng số dòng: {total_rows}"
    )

    print(
        f"Tổng số cột: {total_columns}"
    )

    report = []

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        missing_percentage = (

            missing_count

            / total_rows

            * 100

        )

        data_type = str(
            df[column].dtype
        )

        column_group = classify_column(
            column
        )

        unique_count = int(
            df[column].nunique(
                dropna=True
            )
        )

        report.append({

            "column":
                column,

            "group":
                column_group,

            "data_type":
                data_type,

            "total_rows":
                total_rows,

            "missing_count":
                missing_count,

            "missing_percentage":
                round(
                    missing_percentage,
                    4
                ),

            "non_missing_count":
                total_rows - missing_count,

            "unique_values":
                unique_count

        })

    report_df = pd.DataFrame(
        report
    )

    missing_only_df = report_df[

        report_df[
            "missing_count"
        ] > 0

    ].copy()

    missing_only_df = missing_only_df.sort_values(

        by=[
            "missing_count",
            "missing_percentage"
        ],

        ascending=False

    )

    missing_only_df.to_csv(

        MISSING_ONLY_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    report_df.to_csv(

        MISSING_REPORT_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    summary = (

        report_df

        .groupby(

            "group",

            as_index=False

        )

        .agg({

            "column":
                "count",

            "missing_count":
                "sum"

        })

    )

    summary = summary.rename(

        columns={

            "column":
                "total_columns"

        }

    )

    summary["missing_percentage_total"] = (

        summary[
            "missing_count"
        ]

        / total_rows

        * 100

    )

    columns_with_missing = (

        report_df[

            report_df[
                "missing_count"
            ] > 0

        ]

        .groupby(

            "group"

        )

        .size()

        .reset_index(

            name="columns_with_missing"

        )

    )

    summary = summary.merge(

        columns_with_missing,

        on="group",

        how="left"

    )

    summary[

        "columns_with_missing"

    ] = summary[

        "columns_with_missing"

    ].fillna(

        0

    ).astype(

        int

    )

    summary.to_csv(

        MISSING_SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    total_missing = int(

        report_df[
            "missing_count"
        ].sum()

    )

    total_missing_columns = len(

        missing_only_df

    )

    print()
    print("=" * 70)
    print("MISSING VALUE ANALYSIS COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Tổng số missing values: "
        f"{total_missing}"
    )

    print(
        f"Số cột có missing: "
        f"{total_missing_columns}"
    )

    print()

    for group in [

        "metadata",

        "audio_feature",

        "other"

    ]:

        group_df = report_df[

            report_df[
                "group"
            ] == group

        ]

        group_missing = int(

            group_df[
                "missing_count"
            ].sum()

        )

        group_columns = int(

            (

                group_df[
                    "missing_count"
                ] > 0

            ).sum()

        )

        print(

            f"{group}: "

            f"{group_missing} missing values "

            f"trong {group_columns} cột"

        )

    print()
    print(
        "Top các cột có nhiều missing nhất:"
    )

    print()

    print(

        missing_only_df[

            [

                "column",

                "group",

                "data_type",

                "missing_count",

                "missing_percentage"

            ]

        ]

        .head(20)

        .to_string(

            index=False

        )

    )

    print()
    print(
        "File báo cáo chi tiết:"
    )

    print(
        MISSING_REPORT_PATH
    )

    print()
    print(
        "File chỉ gồm các cột có missing:"
    )

    print(
        MISSING_ONLY_PATH
    )

    print()
    print(
        "File tổng hợp theo nhóm:"
    )

    print(
        MISSING_SUMMARY_PATH
    )

    print()
    print("=" * 70)


if __name__ == "__main__":

    main()