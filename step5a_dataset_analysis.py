import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "processed",
    "step4",
    "audio_features_final.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "processed",
    "step5"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

PARTICIPANT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "participant_summary.csv"
)

SEGMENT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "segment_summary.csv"
)

LABEL_DISTRIBUTION = os.path.join(
    OUTPUT_DIR,
    "label_distribution.csv"
)

PARTICIPANT_DISTRIBUTION = os.path.join(
    OUTPUT_DIR,
    "participant_distribution.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "step5a_report.csv"
)


def find_label_column(df):

    candidates = [
        "label",
        "depression_label",
        "depression",
        "phq_binary",
        "phq8_binary",
        "target",
        "class"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def main():

    print()
    print("=" * 70)
    print("STEP 5A - PARTICIPANT & LABEL ANALYSIS")
    print("=" * 70)

    if not os.path.exists(INPUT_PATH):

        print()
        print("Không tìm thấy file:")
        print(INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)

    if "participant" not in df.columns:

        print()
        print("Không tìm thấy cột participant")
        return

    label_column = find_label_column(df)

    total_segments = len(df)

    total_participants = df["participant"].nunique()

    participant_summary = (
        df.groupby("participant")
        .size()
        .reset_index(name="total_segments")
    )

    if label_column is not None:

        labels = (
            df.groupby("participant")[label_column]
            .first()
            .reset_index()
        )

        participant_summary = participant_summary.merge(
            labels,
            on="participant",
            how="left"
        )

    participant_summary.to_csv(
        PARTICIPANT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    participant_distribution = (
        participant_summary[["participant", "total_segments"]]
    )

    participant_distribution.to_csv(
        PARTICIPANT_DISTRIBUTION,
        index=False,
        encoding="utf-8-sig"
    )

    segment_summary = pd.DataFrame({

        "Metric":[
            "Total Participants",
            "Total Segments",
            "Min Segments",
            "Max Segments",
            "Average Segments",
            "Median Segments"
        ],

        "Value":[

            total_participants,

            total_segments,

            participant_summary["total_segments"].min(),

            participant_summary["total_segments"].max(),

            round(
                participant_summary["total_segments"].mean(),
                2
            ),

            participant_summary["total_segments"].median()

        ]

    })

    segment_summary.to_csv(

        SEGMENT_SUMMARY,

        index=False,

        encoding="utf-8-sig"

    )

    if label_column is not None:

        participant_label = (

            participant_summary

            .groupby(label_column)

            .size()

            .reset_index(name="Participants")

        )

        segment_label = (

            df

            .groupby(label_column)

            .size()

            .reset_index(name="Segments")

        )

        label_distribution = participant_label.merge(

            segment_label,

            on=label_column

        )

    else:

        label_distribution = pd.DataFrame({

            "Information":[

                "Không tìm thấy cột label"

            ]

        })

    label_distribution.to_csv(

        LABEL_DISTRIBUTION,

        index=False,

        encoding="utf-8-sig"

    )

    duplicate_segments = 0

    if "segment_id" in df.columns:

        duplicate_segments = int(

            df["segment_id"].duplicated().sum()

        )

    duplicate_participants = int(

        participant_summary["participant"]

        .duplicated()

        .sum()

    )

    missing_participant = int(

        df["participant"]

        .isna()

        .sum()

    )

    missing_label = 0

    if label_column is not None:

        missing_label = int(

            df[label_column]

            .isna()

            .sum()

        )

    report = pd.DataFrame({

        "Metric":[

            "Total Participants",

            "Total Segments",

            "Total Features",

            "Duplicate Participants",

            "Duplicate Segments",

            "Missing Participants",

            "Missing Labels",

            "Average Segments",

            "Minimum Segments",

            "Maximum Segments",

            "Label Column"

        ],

        "Value":[

            total_participants,

            total_segments,

            len(df.columns),

            duplicate_participants,

            duplicate_segments,

            missing_participant,

            missing_label,

            round(

                participant_summary["total_segments"]

                .mean(),

                2

            ),

            participant_summary["total_segments"]

            .min(),

            participant_summary["total_segments"]

            .max(),

            label_column if label_column else "Not Found"

        ]

    })

    report.to_csv(

        REPORT_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    print()
    print("=" * 70)
    print("STEP 5A COMPLETED")
    print("=" * 70)

    print()

    print(f"Tổng participants: {total_participants}")

    print(f"Tổng segments: {total_segments}")

    print(f"Tổng features: {len(df.columns)}")

    print()

    print(f"Participant ít segment nhất: {participant_summary['total_segments'].min()}")

    print(f"Participant nhiều segment nhất: {participant_summary['total_segments'].max()}")

    print(f"Segment trung bình: {round(participant_summary['total_segments'].mean(),2)}")

    print()

    print(f"Duplicate participant: {duplicate_participants}")

    print(f"Duplicate segment: {duplicate_segments}")

    print(f"Missing participant: {missing_participant}")

    print(f"Missing label: {missing_label}")

    print()

    if label_column:

        print(f"Cột label: {label_column}")

        print()

        print(label_distribution.to_string(index=False))

    else:

        print("Không tìm thấy cột label.")

    print()

    print("Participant summary:")
    print(PARTICIPANT_SUMMARY)

    print()

    print("Segment summary:")
    print(SEGMENT_SUMMARY)

    print()

    print("Label distribution:")
    print(LABEL_DISTRIBUTION)

    print()

    print("Participant distribution:")
    print(PARTICIPANT_DISTRIBUTION)

    print()

    print("Report:")
    print(REPORT_PATH)

    print()

    print("=" * 70)


if __name__ == "__main__":

    main()