import os
import pandas as pd

BASE_DIR = "/Applications/NCKH_DATATEAM"

DATASET_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "step4",
    "audio_features_final.csv"
)

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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed",
    "step5"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "labeled_dataset.csv"
)

LABEL_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step5e_label_report.csv"
)

UNMATCHED_PATH = os.path.join(
    OUTPUT_DIR,
    "step5e_unmatched_participants.csv"
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


def main():

    print("=" * 70)
    print("STEP 5E - MERGE PHQ LABELS")
    print("=" * 70)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy dataset:\n{DATASET_PATH}"
        )

    if not os.path.exists(TRAIN_LABEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy train_split.csv:\n{TRAIN_LABEL_PATH}"
        )

    if not os.path.exists(DEV_LABEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy dev_split.csv:\n{DEV_LABEL_PATH}"
        )

    print("\nĐang đọc dataset...")
    dataset = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"Tổng segment: {len(dataset)}"
    )

    if "participant" not in dataset.columns:
        raise ValueError(
            "Dataset không có cột participant."
        )

    required_label_columns = [
        "Participant_ID",
        "Gender",
        "PHQ_Binary",
        "PHQ_Score"
    ]

    train_labels = pd.read_csv(
        TRAIN_LABEL_PATH
    )

    dev_labels = pd.read_csv(
        DEV_LABEL_PATH
    )

    print(
        f"Train labels: {len(train_labels)} participants"
    )

    print(
        f"Dev labels: {len(dev_labels)} participants"
    )

    missing_train_columns = [
        column
        for column in required_label_columns
        if column not in train_labels.columns
    ]

    missing_dev_columns = [
        column
        for column in required_label_columns
        if column not in dev_labels.columns
    ]

    if missing_train_columns:
        raise ValueError(
            "Train label thiếu cột:\n"
            + "\n".join(
                missing_train_columns
            )
        )

    if missing_dev_columns:
        raise ValueError(
            "Dev label thiếu cột:\n"
            + "\n".join(
                missing_dev_columns
            )
        )

    label_columns = [
        "Participant_ID",
        "Gender",
        "PHQ_Binary",
        "PHQ_Score"
    ]

    train_labels = train_labels[
        label_columns
    ].copy()

    dev_labels = dev_labels[
        label_columns
    ].copy()

    labels = pd.concat(
        [
            train_labels,
            dev_labels
        ],
        ignore_index=True
    )

    labels["participant_key"] = (
        labels["Participant_ID"]
        .apply(
            normalize_participant_id
        )
    )

    dataset["participant_key"] = (
        dataset["participant"]
        .apply(
            normalize_participant_id
        )
    )

    duplicate_label_ids = (
        labels[
            "participant_key"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicate_labels = labels[
        duplicate_label_ids
    ].copy()

    if len(duplicate_labels) > 0:

        duplicate_labels = (
            duplicate_labels
            .sort_values(
                "participant_key"
            )
        )

        print()
        print(
            "⚠️ Phát hiện participant "
            "trùng trong train/dev labels:"
        )

        print(
            duplicate_labels.to_string(
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
        )

    dataset_participants = set(
        dataset[
            "participant_key"
        ].dropna().unique()
    )

    label_participants = set(
        labels[
            "participant_key"
        ].dropna().unique()
    )

    unmatched_dataset = sorted(
        dataset_participants
        - label_participants
    )

    extra_labels = sorted(
        label_participants
        - dataset_participants
    )

    merged = dataset.merge(
        labels,
        on="participant_key",
        how="left",
        suffixes=(
            "",
            "_label"
        )
    )

    merged = merged.drop(
        columns=[
            "participant_key"
        ]
    )

    merged["PHQ_Binary"] = pd.to_numeric(
        merged["PHQ_Binary"],
        errors="coerce"
    )

    merged["PHQ_Score"] = pd.to_numeric(
        merged["PHQ_Score"],
        errors="coerce"
    )

    total_segments = len(
        merged
    )

    matched_segments = int(
        merged[
            "PHQ_Binary"
        ].notna().sum()
    )

    missing_label_segments = int(
        merged[
            "PHQ_Binary"
        ].isna().sum()
    )

    total_participants = (
        merged["participant"]
        .nunique()
    )

    participants_with_label = (
        merged[
            merged[
                "PHQ_Binary"
            ].notna()
        ]["participant"]
        .nunique()
    )

    participants_without_label = (
        merged[
            merged[
                "PHQ_Binary"
            ].isna()
        ]["participant"]
        .nunique()
    )

    duplicate_segments = int(
        merged[
            "segment_id"
        ].duplicated().sum()
    )

    label_distribution = (
        merged[
            [
                "participant",
                "PHQ_Binary",
                "PHQ_Score"
            ]
        ]
        .drop_duplicates(
            subset=[
                "participant"
            ]
        )
        .groupby(
            "PHQ_Binary",
            dropna=False
        )
        .agg(
            participants=(
                "participant",
                "count"
            ),
            mean_phq_score=(
                "PHQ_Score",
                "mean"
            )
        )
        .reset_index()
    )

    segment_distribution = (
        merged[
            "PHQ_Binary"
        ]
        .value_counts(
            dropna=False
        )
        .reset_index()
    )

    segment_distribution.columns = [
        "PHQ_Binary",
        "segments"
    ]

    output_columns = list(
        dataset.columns.drop(
            "participant_key"
        )
    )

    output_columns.extend(
        [
            "Gender",
            "PHQ_Binary",
            "PHQ_Score"
        ]
    )

    merged = merged[
        [
            column
            for column in output_columns
            if column in merged.columns
        ]
    ]

    merged.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    report = pd.DataFrame({

        "metric": [

            "Total segments",

            "Matched segments",

            "Missing label segments",

            "Total participants",

            "Participants with label",

            "Participants without label",

            "Train label participants",

            "Dev label participants",

            "Unique label participants",

            "Dataset participants without label",

            "Extra label participants",

            "Duplicate segment IDs"

        ],

        "value": [

            total_segments,

            matched_segments,

            missing_label_segments,

            total_participants,

            participants_with_label,

            participants_without_label,

            train_labels[
                "Participant_ID"
            ].nunique(),

            dev_labels[
                "Participant_ID"
            ].nunique(),

            labels[
                "participant_key"
            ].nunique(),

            len(
                unmatched_dataset
            ),

            len(
                extra_labels
            ),

            duplicate_segments

        ]

    })

    report.to_csv(
        LABEL_REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    if len(unmatched_dataset) > 0:

        unmatched_df = pd.DataFrame({

            "participant":

                unmatched_dataset

        })

        unmatched_df.to_csv(
            UNMATCHED_PATH,
            index=False,
            encoding="utf-8-sig"
        )

    print()
    print("=" * 70)
    print("STEP 5E COMPLETED")
    print("=" * 70)

    print(
        f"\nTổng segments: {total_segments}"
    )

    print(
        f"Segments match label: {matched_segments}"
    )

    print(
        f"Segments thiếu label: {missing_label_segments}"
    )

    print(
        f"Tổng participants: {total_participants}"
    )

    print(
        f"Participants có label: "
        f"{participants_with_label}"
    )

    print(
        f"Participants thiếu label: "
        f"{participants_without_label}"
    )

    print(
        f"Participant dataset không match: "
        f"{len(unmatched_dataset)}"
    )

    print(
        f"Participant label không có trong dataset: "
        f"{len(extra_labels)}"
    )

    print(
        f"Duplicate segment ID: "
        f"{duplicate_segments}"
    )

    print()
    print(
        "Label distribution:"
    )

    print(
        label_distribution.to_string(
            index=False
        )
    )

    print()
    print(
        "Segment distribution:"
    )

    print(
        segment_distribution.to_string(
            index=False
        )
    )

    print()
    print(
        "Dataset có label:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Báo cáo label:"
    )

    print(
        LABEL_REPORT_PATH
    )

    if len(unmatched_dataset) > 0:

        print()
        print(
            "Participant chưa match:"
        )

        print(
            UNMATCHED_PATH
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()