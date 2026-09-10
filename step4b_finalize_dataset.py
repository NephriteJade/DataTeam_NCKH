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

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "audio_features_final.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step4_final_quality_report.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def main():

    print()
    print("=" * 70)
    print("STEP 4B - FINALIZE CLEAN DATASET")
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

    original_rows = len(df)

    original_columns = len(
        df.columns
    )

    print()
    print(
        f"Số dòng ban đầu: "
        f"{original_rows}"
    )

    print(
        f"Số cột ban đầu: "
        f"{original_columns}"
    )

    columns_to_remove = [

        "audio_error",

        "feature_error"

    ]

    removed_columns = [

        column

        for column in columns_to_remove

        if column in df.columns

    ]

    if len(removed_columns) > 0:

        df = df.drop(

            columns=removed_columns

        )

    final_rows = len(df)

    final_columns = len(
        df.columns
    )

    missing_total = int(

        df.isna().sum().sum()

    )

    numeric_columns = (

        df.select_dtypes(

            include=[np.number]

        ).columns.tolist()

    )

    inf_total = 0

    for column in numeric_columns:

        inf_total += int(

            np.isinf(

                df[column].to_numpy()

            ).sum()

        )

    duplicate_rows = int(

        df.duplicated().sum()

    )

    id_columns = [

        column

        for column in df.columns

        if (

            "participant_id" in column.lower()

            or "participant" in column.lower()

            or "subject_id" in column.lower()

            or "subject" in column.lower()

            or "segment_id" in column.lower()

            or "segment" in column.lower()

        )

    ]

    id_report = []

    for column in id_columns:

        id_report.append({

            "column":
                column,

            "total_rows":
                final_rows,

            "unique_values":
                int(

                    df[column].nunique(

                        dropna=True

                    )

                ),

            "missing_values":
                int(

                    df[column].isna().sum()

                ),

            "duplicate_values":
                int(

                    df[column].duplicated().sum()

                )

        })

    id_report_df = pd.DataFrame(

        id_report

    )

    df.to_csv(

        OUTPUT_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    quality_report = pd.DataFrame({

        "metric": [

            "original_rows",

            "original_columns",

            "final_rows",

            "final_columns",

            "removed_error_columns",

            "missing_values",

            "inf_values",

            "duplicate_rows",

            "numeric_columns",

            "id_columns"

        ],

        "value": [

            original_rows,

            original_columns,

            final_rows,

            final_columns,

            ", ".join(

                removed_columns

            ),

            missing_total,

            inf_total,

            duplicate_rows,

            len(

                numeric_columns

            ),

            ", ".join(

                id_columns

            )

        ]

    })

    quality_report.to_csv(

        REPORT_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    if len(id_report_df) > 0:

        id_report_df.to_csv(

            os.path.join(

                OUTPUT_DIR,

                "step4_final_id_report.csv"

            ),

            index=False,

            encoding="utf-8-sig"

        )

    print()
    print("=" * 70)
    print("STEP 4B COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Số dòng ban đầu: "
        f"{original_rows}"
    )

    print(
        f"Số cột ban đầu: "
        f"{original_columns}"
    )

    print(
        f"Số dòng cuối: "
        f"{final_rows}"
    )

    print(
        f"Số cột cuối: "
        f"{final_columns}"
    )

    print()
    print(
        "Các cột error đã xóa:"
    )

    if len(removed_columns) > 0:

        for column in removed_columns:

            print(
                f"- {column}"
            )

    else:

        print(
            "Không có cột error cần xóa."
        )

    print()
    print(
        f"Missing values còn lại: "
        f"{missing_total}"
    )

    print(
        f"Inf values còn lại: "
        f"{inf_total}"
    )

    print(
        f"Dòng trùng lặp: "
        f"{duplicate_rows}"
    )

    print()
    print(
        "Các cột ID được phát hiện:"
    )

    if len(id_columns) > 0:

        for column in id_columns:

            print(
                f"- {column}"
            )

    else:

        print(
            "Không phát hiện cột ID."
        )

    print()
    print(
        "Dataset cuối:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Báo cáo chất lượng:"
    )

    print(
        REPORT_PATH
    )

    print()
    print("=" * 70)


if __name__ == "__main__":

    main()