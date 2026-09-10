import os
import pandas as pd

BASE_DIR = "/Applications/NCKH_DATATEAM"

CLASS_0_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "step5",
    "step5g_missing_class_0.csv"
)

CLASS_1_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "step5",
    "step5g_missing_class_1.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed",
    "step5"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "step5h_selected_participants.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "step5h_summary.csv"
)

CLASS_0_COUNT = 15
CLASS_1_COUNT = 15

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def select_participants(
    path,
    label,
    count
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Không tìm thấy file:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    if len(df) < count:

        raise ValueError(
            f"Không đủ participant cho class {label}. "
            f"Có {len(df)}, cần {count}."
        )

    df = df[
        df["PHQ_Binary"] == label
    ].copy()

    df = df.sort_values(
        [
            "split_source",
            "participant_key"
        ]
    )

    selected = df.head(
        count
    ).copy()

    return selected


def main():

    print("=" * 70)
    print("STEP 5H - SELECT EXPANDED PARTICIPANTS")
    print("=" * 70)

    class_0 = select_participants(
        CLASS_0_PATH,
        0,
        CLASS_0_COUNT
    )

    class_1 = select_participants(
        CLASS_1_PATH,
        1,
        CLASS_1_COUNT
    )

    selected = pd.concat(
        [
            class_0,
            class_1
        ],
        ignore_index=True
    )

    selected = selected.sort_values(
        [
            "PHQ_Binary",
            "split_source",
            "participant_key"
        ]
    ).reset_index(
        drop=True
    )

    selected["selection_order"] = (
        range(
            1,
            len(selected) + 1
        )
    )

    output_columns = [
        "selection_order",
        "Participant_ID",
        "participant_key",
        "dataset_folder",
        "Gender",
        "PHQ_Binary",
        "PHQ_Score",
        "split_source"
    ]

    selected = selected[
        [
            column
            for column in output_columns
            if column in selected.columns
        ]
    ]

    selected.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    summary = (
        selected
        .groupby(
            [
                "PHQ_Binary",
                "split_source"
            ],
            as_index=False
        )
        .agg(
            participants=(
                "participant_key",
                "nunique"
            ),
            mean_phq_score=(
                "PHQ_Score",
                "mean"
            ),
            min_phq_score=(
                "PHQ_Score",
                "min"
            ),
            max_phq_score=(
                "PHQ_Score",
                "max"
            )
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("STEP 5H COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Tổng participant được chọn: "
        f"{len(selected)}"
    )

    print(
        f"PHQ_Binary = 0: "
        f"{len(class_0)}"
    )

    print(
        f"PHQ_Binary = 1: "
        f"{len(class_1)}"
    )

    print()
    print(
        "Phân bố theo split:"
    )

    print(
        selected[
            [
                "PHQ_Binary",
                "split_source"
            ]
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Danh sách participant được chọn:"
    )

    print(
        selected[
            [
                "Participant_ID",
                "PHQ_Binary",
                "PHQ_Score",
                "split_source"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "File danh sách participant:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "File summary:"
    )

    print(
        SUMMARY_PATH
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()