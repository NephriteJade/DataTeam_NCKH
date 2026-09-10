import os
import numpy as np
import pandas as pd

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

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def main():

    print()
    print("=" * 70)
    print("STEP 4 - DATA QUALITY CHECK & FEATURE PROCESSING")
    print("=" * 70)

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

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    non_numeric_columns = df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

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
                missing_percentage

        })

    missing_df = pd.DataFrame(
        missing_report
    )

    inf_report = []

    for column in numeric_columns:

        inf_count = int(
            np.isinf(
                df[column].to_numpy()
            ).sum()
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

    if len(inf_columns) > 0:

        for column in inf_columns:

            df[column] = df[column].replace(

                [np.inf, -np.inf],

                np.nan

            )

    feature_status_columns = [

        "feature_status",

        "feature_error",

        "status"

    ]

    metadata_keywords = [

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

        "error"

    ]

    metadata_columns = []

    for column in df.columns:

        column_lower = column.lower()

        if (

            column in feature_status_columns

            or any(

                keyword in column_lower

                for keyword in metadata_keywords

            )

        ):

            metadata_columns.append(
                column
            )

    feature_columns = [

        column

        for column in numeric_columns

        if column not in metadata_columns

    ]

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

        df = df.drop(

            columns=zero_variance_columns

        )

    remaining_numeric_columns = (

        df.select_dtypes(

            include=[np.number]

        ).columns.tolist()

    )

    for column in remaining_numeric_columns:

        if df[column].isna().any():

            median_value = df[column].median()

            df[column] = df[column].fillna(

                median_value

            )

    final_missing_count = int(

        df.isna().sum().sum()

    )

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

    duplicate_rows = int(

        df.duplicated().sum()

    )

    duplicate_id_report = pd.DataFrame()

    possible_id_columns = [

        column

        for column in df.columns

        if (

            "participant_id" in column.lower()

            or "subject_id" in column.lower()

            or "segment_id" in column.lower()

            or column.lower() == "participant"

            or column.lower() == "subject"

        )

    ]

    if len(possible_id_columns) > 0:

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

            len(zero_variance_columns),

            duplicate_rows,

            final_missing_count,

            final_inf_count

        ]

    })

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

            os.path.join(

                OUTPUT_DIR,

                "step4_id_quality_report.csv"

            ),

            index=False,

            encoding="utf-8-sig"

        )

    print()
    print("=" * 70)
    print("STEP 4 COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Số dòng ban đầu: {original_rows}"
    )

    print(
        f"Số cột ban đầu: {original_columns}"
    )

    print(
        f"Số dòng sau xử lý: {len(df)}"
    )

    print(
        f"Số cột sau xử lý: {len(df.columns)}"
    )

    print(
        f"Số cột Inf được xử lý: {len(inf_columns)}"
    )

    print(
        f"Số cột phương sai bằng 0 đã loại bỏ: "
        f"{len(zero_variance_columns)}"
    )

    print(
        f"Số dòng trùng lặp: {duplicate_rows}"
    )

    print(
        f"Missing values còn lại: "
        f"{final_missing_count}"
    )

    print(
        f"Inf values còn lại: "
        f"{final_inf_count}"
    )

    print()
    print("File dữ liệu sau xử lý:")
    print(OUTPUT_PATH)

    print()
    print("File báo cáo chất lượng:")
    print(REPORT_PATH)

    if len(duplicate_id_report) > 0:

        print()
        print("File kiểm tra ID:")
        print(

            os.path.join(

                OUTPUT_DIR,

                "step4_id_quality_report.csv"

            )

        )

    print()
    print("=" * 70)


if __name__ == "__main__":

    main()