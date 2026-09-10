import os
import pandas as pd

BASE_DIR = "/Applications/NCKH_DATATEAM"

TRAIN_LABEL_PATH = os.path.join(
    BASE_DIR,
    "labels",
    "train_split.csv"
)

DEV_LABEL_PATH = os.path.join(
    BASE_DIR,
    "labels",
    "dev_split.csv"
)

EDAIC_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "E-DAIC"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed",
    "step5"
)

ALL_LABELED_PATH = os.path.join(
    OUTPUT_DIR,
    "step5f_all_labeled_participants.csv"
)

LABEL_DISTRIBUTION_PATH = os.path.join(
    OUTPUT_DIR,
    "step5f_label_distribution.csv"
)

AVAILABLE_PATH = os.path.join(
    OUTPUT_DIR,
    "step5f_available_participants.csv"
)

MISSING_PATH = os.path.join(
    OUTPUT_DIR,
    "step5f_missing_participants.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "step5f_summary.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def normalize_participant_id(value):

    if pd.isna(value):
        return None

    value = str(
        value
    ).strip()

    if value.endswith("_P"):
        value = value[:-2]

    return value


def get_existing_participants():

    if not os.path.exists(
        EDAIC_DIR
    ):
        return set()

    participants = set()

    for item in os.listdir(
        EDAIC_DIR
    ):

        item_path = os.path.join(
            EDAIC_DIR,
            item
        )

        if not os.path.isdir(
            item_path
        ):
            continue

        normalized = normalize_participant_id(
            item
        )

        if normalized is not None:
            participants.add(
                normalized
            )

    return participants


def main():

    print("=" * 70)
    print("STEP 5F - EXPAND DATASET ANALYSIS")
    print("=" * 70)

    if not os.path.exists(
        TRAIN_LABEL_PATH
    ):
        raise FileNotFoundError(
            f"Không tìm thấy:\n{TRAIN_LABEL_PATH}"
        )

    if not os.path.exists(
        DEV_LABEL_PATH
    ):
        raise FileNotFoundError(
            f"Không tìm thấy:\n{DEV_LABEL_PATH}"
        )

    print()
    print("Đọc train_split.csv...")

    train_df = pd.read_csv(
        TRAIN_LABEL_PATH
    )

    print(
        f"Train participants: "
        f"{len(train_df)}"
    )

    print()
    print("Đọc dev_split.csv...")

    dev_df = pd.read_csv(
        DEV_LABEL_PATH
    )

    print(
        f"Dev participants: "
        f"{len(dev_df)}"
    )

    required_columns = [
        "Participant_ID",
        "Gender",
        "PHQ_Binary",
        "PHQ_Score"
    ]

    for column in required_columns:

        if column not in train_df.columns:
            raise ValueError(
                f"Train thiếu cột: {column}"
            )

        if column not in dev_df.columns:
            raise ValueError(
                f"Dev thiếu cột: {column}"
            )

    train_df = train_df[
        required_columns
    ].copy()

    dev_df = dev_df[
        required_columns
    ].copy()

    train_df["split_source"] = "train"
    dev_df["split_source"] = "dev"

    labels = pd.concat(
        [
            train_df,
            dev_df
        ],
        ignore_index=True
    )

    labels["participant_key"] = (
        labels["Participant_ID"]
        .apply(
            normalize_participant_id
        )
    )

    duplicate_ids = (
        labels[
            "participant_key"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicate_df = labels[
        duplicate_ids
    ].copy()

    if len(duplicate_df) > 0:

        print()
        print(
            "⚠️ Participant xuất hiện nhiều lần:"
        )

        print(
            duplicate_df[
                [
                    "Participant_ID",
                    "split_source",
                    "PHQ_Binary",
                    "PHQ_Score"
                ]
            ].to_string(
                index=False
            )
        )

    labels = (
        labels
        .drop_duplicates(
            subset=[
                "participant_key"
            ],
            keep="first"
        )
        .copy()
    )

    existing_participants = (
        get_existing_participants()
    )

    labels["dataset_available"] = (
        labels["participant_key"]
        .isin(
            existing_participants
        )
    )

    labels["dataset_folder"] = (
        labels["participant_key"]
        .apply(
            lambda x:
            f"{x}_P"
            if pd.notna(x)
            else ""
        )
    )

    labels["status"] = (
        labels["dataset_available"]
        .map(
            {
                True: "available",
                False: "missing"
            }
        )
    )

    all_labeled = labels[
        [
            "Participant_ID",
            "participant_key",
            "Gender",
            "PHQ_Binary",
            "PHQ_Score",
            "split_source",
            "dataset_available",
            "dataset_folder",
            "status"
        ]
    ].copy()

    all_labeled.to_csv(
        ALL_LABELED_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    label_distribution = (
        all_labeled
        .groupby(
            "PHQ_Binary"
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
        .reset_index()
    )

    label_distribution.to_csv(
        LABEL_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    available_df = all_labeled[
        all_labeled[
            "dataset_available"
        ]
        == True
    ].copy()

    available_df.to_csv(
        AVAILABLE_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    missing_df = all_labeled[
        all_labeled[
            "dataset_available"
        ]
        == False
    ].copy()

    missing_df.to_csv(
        MISSING_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    train_count = int(
        (
            all_labeled[
                "split_source"
            ]
            == "train"
        ).sum()
    )

    dev_count = int(
        (
            all_labeled[
                "split_source"
            ]
            == "dev"
        ).sum()
    )

    total_labeled = len(
        all_labeled
    )

    available_count = len(
        available_df
    )

    missing_count = len(
        missing_df
    )

    class_zero_count = int(
        (
            all_labeled[
                "PHQ_Binary"
            ]
            == 0
        ).sum()
    )

    class_one_count = int(
        (
            all_labeled[
                "PHQ_Binary"
            ]
            == 1
        ).sum()
    )

    summary = pd.DataFrame({

        "metric": [

            "Total labeled participants",

            "Train participants",

            "Dev participants",

            "Dataset participants available",

            "Dataset participants missing",

            "PHQ_Binary = 0",

            "PHQ_Binary = 1",

            "Duplicate label records",

            "Current dataset participant count"

        ],

        "value": [

            total_labeled,

            train_count,

            dev_count,

            available_count,

            missing_count,

            class_zero_count,

            class_one_count,

            len(
                duplicate_df
            ),

            len(
                existing_participants
            )

        ]

    })

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("STEP 5F COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Tổng labeled participants: "
        f"{total_labeled}"
    )

    print(
        f"Train participants: "
        f"{train_count}"
    )

    print(
        f"Dev participants: "
        f"{dev_count}"
    )

    print(
        f"Participant đã có dataset: "
        f"{available_count}"
    )

    print(
        f"Participant chưa có dataset: "
        f"{missing_count}"
    )

    print()
    print(
        f"PHQ_Binary = 0: "
        f"{class_zero_count}"
    )

    print(
        f"PHQ_Binary = 1: "
        f"{class_one_count}"
    )

    print()
    print(
        "Phân bố label:"
    )

    print(
        label_distribution.to_string(
            index=False
        )
    )

    print()
    print(
        "File toàn bộ labeled participants:"
    )

    print(
        ALL_LABELED_PATH
    )

    print()
    print(
        "File label distribution:"
    )

    print(
        LABEL_DISTRIBUTION_PATH
    )

    print()
    print(
        "File participant đã có dataset:"
    )

    print(
        AVAILABLE_PATH
    )

    print()
    print(
        "File participant cần bổ sung:"
    )

    print(
        MISSING_PATH
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