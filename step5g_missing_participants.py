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

ALL_MISSING_PATH = os.path.join(
    OUTPUT_DIR,
    "step5g_missing_participants.csv"
)

CLASS_0_PATH = os.path.join(
    OUTPUT_DIR,
    "step5g_missing_class_0.csv"
)

CLASS_1_PATH = os.path.join(
    OUTPUT_DIR,
    "step5g_missing_class_1.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "step5g_summary.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def normalize_participant_id(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith("_P"):
        value = value[:-2]

    return value


def get_existing_participants():

    existing = set()

    if not os.path.exists(EDAIC_DIR):
        return existing

    for item in os.listdir(EDAIC_DIR):

        item_path = os.path.join(
            EDAIC_DIR,
            item
        )

        if not os.path.isdir(item_path):
            continue

        participant_id = normalize_participant_id(
            item
        )

        if participant_id is not None:
            existing.add(
                participant_id
            )

    return existing


def load_labels(path, split_name):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy file:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "Participant_ID",
        "Gender",
        "PHQ_Binary",
        "PHQ_Score"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name} thiếu cột:\n"
            + "\n".join(missing_columns)
        )

    df = df[
        required_columns
    ].copy()

    df["split_source"] = split_name

    df["participant_key"] = (
        df["Participant_ID"]
        .apply(
            normalize_participant_id
        )
    )

    return df


def main():

    print("=" * 70)
    print("STEP 5G - MISSING PARTICIPANT ANALYSIS")
    print("=" * 70)

    train_df = load_labels(
        TRAIN_LABEL_PATH,
        "train"
    )

    dev_df = load_labels(
        DEV_LABEL_PATH,
        "dev"
    )

    labels = pd.concat(
        [
            train_df,
            dev_df
        ],
        ignore_index=True
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

    missing_df = labels[
        labels["dataset_available"] == False
    ].copy()

    missing_df["dataset_folder"] = (
        missing_df["participant_key"]
        .apply(
            lambda x:
            f"{x}_P"
        )
    )

    missing_df["class"] = (
        missing_df["PHQ_Binary"]
        .astype("Int64")
    )

    missing_df = missing_df[
        [
            "Participant_ID",
            "participant_key",
            "dataset_folder",
            "Gender",
            "PHQ_Binary",
            "PHQ_Score",
            "split_source"
        ]
    ].sort_values(
        [
            "PHQ_Binary",
            "participant_key"
        ]
    )

    class_0_df = missing_df[
        missing_df["PHQ_Binary"] == 0
    ].copy()

    class_1_df = missing_df[
        missing_df["PHQ_Binary"] == 1
    ].copy()

    missing_df.to_csv(
        ALL_MISSING_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    class_0_df.to_csv(
        CLASS_0_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    class_1_df.to_csv(
        CLASS_1_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    train_missing = int(
        (
            missing_df["split_source"]
            == "train"
        ).sum()
    )

    dev_missing = int(
        (
            missing_df["split_source"]
            == "dev"
        ).sum()
    )

    class_0_count = len(
        class_0_df
    )

    class_1_count = len(
        class_1_df
    )

    summary = pd.DataFrame({

        "metric": [

            "Total labeled participants",

            "Current E-DAIC participants",

            "Missing participants",

            "Missing train participants",

            "Missing dev participants",

            "Missing PHQ_Binary = 0",

            "Missing PHQ_Binary = 1"

        ],

        "value": [

            len(labels),

            len(existing_participants),

            len(missing_df),

            train_missing,

            dev_missing,

            class_0_count,

            class_1_count

        ]

    })

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("STEP 5G COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Tổng labeled participants: "
        f"{len(labels)}"
    )

    print(
        f"Participant đã có dataset: "
        f"{len(existing_participants)}"
    )

    print(
        f"Participant cần bổ sung: "
        f"{len(missing_df)}"
    )

    print(
        f"Missing train: "
        f"{train_missing}"
    )

    print(
        f"Missing dev: "
        f"{dev_missing}"
    )

    print()
    print(
        f"Missing PHQ_Binary = 0: "
        f"{class_0_count}"
    )

    print(
        f"Missing PHQ_Binary = 1: "
        f"{class_1_count}"
    )

    print()
    print(
        "File toàn bộ participant cần bổ sung:"
    )

    print(
        ALL_MISSING_PATH
    )

    print()
    print(
        "File class 0:"
    )

    print(
        CLASS_0_PATH
    )

    print()
    print(
        "File class 1:"
    )

    print(
        CLASS_1_PATH
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