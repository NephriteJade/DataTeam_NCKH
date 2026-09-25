import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

METADATA_PATH = PROJECT_ROOT / "processed" / "metadata.csv"

STEP3_DIR = PROJECT_ROOT / "processed" / "step3"

AUDIO_SEGMENTS_DIR = STEP3_DIR / "audio_segments"
AUDIO_METADATA_PATH = STEP3_DIR / "audio_metadata.csv"
AUDIO_FEATURES_PATH = STEP3_DIR / "audio_features.csv"

# Audio parameters
TARGET_SR = 16000
MONO = True
NORMALIZE_AUDIO = True

# Feature parameters
N_MFCC = 13
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


STEP3_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):
    """
    Tìm tên cột bất kể viết hoa/thường, khoảng trắng, dấu '-'.
    """

    normalized_columns = {
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_"): col
        for col in df.columns
    }

    for name in possible_names:
        norm_name = (
            name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if norm_name in normalized_columns:
            return normalized_columns[norm_name]

    return None


def is_valid_number(value):
    """
    Kiểm tra value có phải số hợp lệ hay không.
    """

    if value is None:
        return False

    try:
        value = float(value)

        if np.isnan(value):
            return False

        if np.isinf(value):
            return False

        return True

    except (TypeError, ValueError):
        return False


# ============================================================
# AUDIO PATH
# ============================================================

def resolve_audio_path(audio_path):
    """
    Tìm file audio thực tế từ audio_path trong metadata.
    """

    if audio_path is None or pd.isna(audio_path):
        return None

    audio_str = str(audio_path).strip()

    if not audio_str:
        return None

    if audio_str.lower() in [
        "not_available",
        "none",
        "nan",
        "null"
    ]:
        return None

    path = Path(audio_str)

    candidates = [
        path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "dataset" / path,
        PROJECT_ROOT / "dataset" / "E-DAIC" / path,
    ]

    for cand in candidates:

        try:
            if cand.exists() and cand.is_file():
                return cand.resolve()
        except Exception:
            pass

    return None


# ============================================================
# TRANSCRIPT PATH
# ============================================================

def find_transcript_file(audio_path):
    """
    Từ:

        dataset/E-DAIC/667_P/667_AUDIO.wav

    tìm:

        dataset/E-DAIC/667_P/667_Transcript.csv
    """

    audio_file = resolve_audio_path(audio_path)

    if audio_file is None:
        return None

    participant_dir = audio_file.parent

    # Ví dụ:
    # 667_AUDIO.wav -> 667_Transcript.csv

    audio_stem = audio_file.stem

    if "_AUDIO" in audio_stem.upper():
        participant_number = audio_stem.split("_")[0]

        transcript_path = (
            participant_dir /
            f"{participant_number}_Transcript.csv"
        )

        if transcript_path.exists():
            return transcript_path

    # Fallback: tìm mọi file Transcript.csv
    transcript_files = list(
        participant_dir.glob("*_Transcript.csv")
    )

    if len(transcript_files) > 0:
        return transcript_files[0]

    return None


# ============================================================
# GET TIMESTAMP FROM TRANSCRIPT
# ============================================================

def get_timestamp_from_transcript(audio_path, segment_id, text=None):
    """
    Nếu metadata bị thiếu start/end time,
    tìm timestamp trong file Transcript.csv.

    Ví dụ:

        segment_id = 667_P_111

    sẽ tìm row 111 trong:

        667_Transcript.csv

    """

    transcript_path = find_transcript_file(audio_path)

    if transcript_path is None:
        return None, None, None

    try:
        transcript_df = pd.read_csv(transcript_path)

    except Exception as e:
        print(
            f"Không thể đọc transcript "
            f"{transcript_path}: {e}"
        )
        return None, None, None

    # --------------------------------------------------------
    # Tìm column
    # --------------------------------------------------------

    start_col = find_column(
        transcript_df,
        [
            "start_time",
            "start",
            "Start_Time"
        ]
    )

    end_col = find_column(
        transcript_df,
        [
            "end_time",
            "end",
            "End_Time"
        ]
    )

    text_col = find_column(
        transcript_df,
        [
            "text",
            "utterance",
            "sentence"
        ]
    )

    if start_col is None or end_col is None:
        return None, None, None

    # --------------------------------------------------------
    # Lấy số segment
    #
    # Ví dụ:
    # 667_P_111 -> 111
    # --------------------------------------------------------

    try:

        segment_number = int(
            str(segment_id).split("_")[-1]
        )

    except Exception:
        segment_number = None

    # --------------------------------------------------------
    # Cách 1:
    # segment number = index của transcript
    # --------------------------------------------------------

    if (
        segment_number is not None
        and 0 <= segment_number < len(transcript_df)
    ):

        row = transcript_df.iloc[segment_number]

        start_value = row[start_col]
        end_value = row[end_col]

        if (
            is_valid_number(start_value)
            and is_valid_number(end_value)
        ):

            text_value = (
                row[text_col]
                if text_col is not None
                else ""
            )

            return (
                float(start_value),
                float(end_value),
                str(text_value)
            )

    # --------------------------------------------------------
    # Cách 2:
    # Nếu không match được index thì thử tìm text
    # --------------------------------------------------------

    if (
        text_col is not None
        and text is not None
        and not pd.isna(text)
    ):

        target_text = str(text).strip().lower()

        for _, row in transcript_df.iterrows():

            current_text = str(
                row[text_col]
            ).strip().lower()

            if current_text == target_text:

                start_value = row[start_col]
                end_value = row[end_col]

                if (
                    is_valid_number(start_value)
                    and is_valid_number(end_value)
                ):

                    return (
                        float(start_value),
                        float(end_value),
                        current_text
                    )

    return None, None, None


# ============================================================
# LOAD AUDIO SEGMENT
# ============================================================

def load_audio_segment(
    audio_path,
    start_time=None,
    end_time=None,
    segment_id=None,
    text=None
):

    audio_file = resolve_audio_path(audio_path)

    if audio_file is None:
        raise FileNotFoundError(
            f"Không tìm thấy audio: {audio_path}"
        )

    # --------------------------------------------------------
    # 1. Kiểm tra timestamp từ metadata
    # --------------------------------------------------------

    start_valid = is_valid_number(start_time)
    end_valid = is_valid_number(end_time)

    timestamp_source = "metadata"

    # --------------------------------------------------------
    # 2. Nếu metadata bị NaN -> tìm Transcript.csv
    # --------------------------------------------------------

    if not (start_valid and end_valid):

        (
            transcript_start,
            transcript_end,
            transcript_text
        ) = get_timestamp_from_transcript(
            audio_path,
            segment_id,
            text
        )

        if (
            is_valid_number(transcript_start)
            and is_valid_number(transcript_end)
        ):

            start_time = transcript_start
            end_time = transcript_end

            timestamp_source = "transcript"

        else:

            raise ValueError(
                f"Không tìm được timestamp cho segment "
                f"{segment_id}. "
                f"start={start_time}, end={end_time}"
            )

    # --------------------------------------------------------
    # 3. Convert timestamp
    # --------------------------------------------------------

    start = float(start_time)
    end = float(end_time)

    if start < 0:
        raise ValueError(
            f"Start time < 0: {start}"
        )

    if end <= start:
        raise ValueError(
            f"Timestamp không hợp lệ: "
            f"{start} -> {end}"
        )

    duration = end - start

    # --------------------------------------------------------
    # 4. Load đúng đoạn audio
    # --------------------------------------------------------

    y, sr = librosa.load(
        str(audio_file),
        sr=TARGET_SR,
        mono=MONO,
        offset=start,
        duration=duration
    )

    if len(y) == 0:
        raise ValueError(
            f"Audio segment rỗng: "
            f"{start} -> {end}"
        )

    return (
        y,
        sr,
        audio_file,
        start,
        end,
        timestamp_source
    )


# ============================================================
# STEP 3A
# AUDIO SEGMENTATION
# ============================================================

def run_step_3a():

    print(
        "\n"
        + "=" * 70
        + "\nSTEP 3A - AUDIO SEGMENTATION\n"
        + "=" * 70
    )

    if not METADATA_PATH.exists():

        print(
            f"❌ Không tìm thấy metadata.csv tại:\n"
            f"{METADATA_PATH}"
        )

        return False

    df = pd.read_csv(METADATA_PATH)

    print(
        f"📄 Đọc metadata: {len(df)} records"
    )

    # --------------------------------------------------------
    # Detect columns
    # --------------------------------------------------------

    part_col = find_column(
        df,
        [
            "participant_id",
            "participant",
            "participantid"
        ]
    )

    seg_col = find_column(
        df,
        [
            "segment_id",
            "segment",
            "segmentid"
        ]
    )

    audio_col = find_column(
        df,
        [
            "audio_path",
            "audio",
            "audio_file"
        ]
    )

    start_col = find_column(
        df,
        [
            "start_time",
            "start"
        ]
    )

    end_col = find_column(
        df,
        [
            "end_time",
            "end"
        ]
    )

    text_col = find_column(
        df,
        [
            "text",
            "utterance",
            "sentence"
        ]
    )

    print("\nDetected columns:")

    print(
        f"  participant = {part_col}"
    )

    print(
        f"  segment     = {seg_col}"
    )

    print(
        f"  audio       = {audio_col}"
    )

    print(
        f"  start       = {start_col}"
    )

    print(
        f"  end         = {end_col}"
    )

    print(
        f"  text        = {text_col}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    metadata_timestamp_count = 0
    transcript_timestamp_count = 0

    results = []
    error_samples = []

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing Audio"
    ):

        record = row.to_dict()

        # ----------------------------------------------------
        # Participant / Segment
        # ----------------------------------------------------

        p_id = (
            str(row[part_col])
            if part_col is not None
            else "unknown_p"
        )

        s_id = (
            str(row[seg_col])
            if seg_col is not None
            else f"seg_{idx}"
        )

        text_value = (
            row[text_col]
            if text_col is not None
            else ""
        )

        safe_p = (
            p_id
            .replace("/", "_")
            .replace("\\", "_")
        )

        safe_s = (
            s_id
            .replace("/", "_")
            .replace("\\", "_")
        )

        out_file = (
            AUDIO_SEGMENTS_DIR /
            f"{safe_p}_{safe_s}.wav"
        )

        # ----------------------------------------------------
        # Original timestamp
        # ----------------------------------------------------

        start_value = (
            row[start_col]
            if start_col is not None
            else None
        )

        end_value = (
            row[end_col]
            if end_col is not None
            else None
        )

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        try:

            (
                y,
                sr,
                resolved_audio,
                actual_start,
                actual_end,
                timestamp_source
            ) = load_audio_segment(
                row[audio_col],
                start_value,
                end_value,
                segment_id=s_id,
                text=text_value
            )

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            if NORMALIZE_AUDIO:

                max_amplitude = np.max(
                    np.abs(y)
                )

                if max_amplitude > 0:

                    y = (
                        y /
                        max_amplitude
                    )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            sf.write(
                str(out_file),
                y,
                sr
            )

            rel_out_path = (
                out_file.relative_to(
                    PROJECT_ROOT
                )
            )

            # ------------------------------------------------
            # Count timestamp source
            # ------------------------------------------------

            if timestamp_source == "metadata":
                metadata_timestamp_count += 1

            elif timestamp_source == "transcript":
                transcript_timestamp_count += 1

            # ------------------------------------------------
            # Update record
            # ------------------------------------------------

            record.update({

                "actual_start_time":
                    float(actual_start),

                "actual_end_time":
                    float(actual_end),

                "timestamp_source":
                    timestamp_source,

                "processed_audio_path":
                    str(rel_out_path),

                "resolved_original_audio_path":
                    str(resolved_audio),

                "audio_duration":
                    float(len(y) / sr),

                "audio_sampling_rate":
                    int(sr),

                "audio_status":
                    "success",

                "audio_error":
                    ""

            })

        except Exception as e:

            if len(error_samples) < 10:

                error_samples.append(
                    (
                        idx,
                        str(
                            row.get(audio_col)
                        ),
                        s_id,
                        str(e)
                    )
                )

            record.update({

                "processed_audio_path":
                    "",

                "resolved_original_audio_path":
                    "",

                "actual_start_time":
                    np.nan,

                "actual_end_time":
                    np.nan,

                "timestamp_source":
                    "",

                "audio_duration":
                    np.nan,

                "audio_sampling_rate":
                    np.nan,

                "audio_status":
                    "error",

                "audio_error":
                    str(e)

            })

        results.append(record)

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    out_df = pd.DataFrame(results)

    out_df.to_csv(
        AUDIO_METADATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    success_cnt = int(
        (
            out_df["audio_status"]
            == "success"
        ).sum()
    )

    error_cnt = int(
        (
            out_df["audio_status"]
            == "error"
        ).sum()
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "✅ STEP 3A HOÀN THÀNH"
    )

    print(
        f"   Thành công : {success_cnt}"
    )

    print(
        f"   Lỗi        : {error_cnt}"
    )

    print(
        f"   Timestamp từ metadata  : "
        f"{metadata_timestamp_count}"
    )

    print(
        f"   Timestamp từ transcript: "
        f"{transcript_timestamp_count}"
    )

    print(
        f"   Output: {AUDIO_METADATA_PATH}"
    )

    # --------------------------------------------------------
    # Error samples
    # --------------------------------------------------------

    if error_cnt > 0:

        print(
            "\n⚠️ CÁC LỖI MẪU:"
        )

        for (
            idx,
            orig_path,
            segment_id,
            err
        ) in error_samples:

            print(
                f" - Row {idx} | "
                f"segment={segment_id} | "
                f"audio='{orig_path}' | "
                f"Lỗi: {err}"
            )

    print(
        "=" * 70
    )

    return success_cnt > 0


# ============================================================
# FEATURE STATISTICS
# ============================================================

def calculate_statistics(values, prefix):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return {

            f"{prefix}_mean":
                np.nan,

            f"{prefix}_std":
                np.nan,

            f"{prefix}_min":
                np.nan,

            f"{prefix}_max":
                np.nan
        }

    return {

        f"{prefix}_mean":
            float(np.mean(values)),

        f"{prefix}_std":
            float(np.std(values)),

        f"{prefix}_min":
            float(np.min(values)),

        f"{prefix}_max":
            float(np.max(values))
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_all_features(audio_path):

    y, sr = librosa.load(
        str(audio_path),
        sr=TARGET_SR,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("Audio empty")

    features = {

        "feature_audio_duration":
            float(len(y) / sr),

        "feature_sampling_rate":
            int(sr)

    }

    # ========================================================
    # MFCC
    # ========================================================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    delta = librosa.feature.delta(
        mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    for i in range(N_MFCC):

        features.update(
            calculate_statistics(
                mfcc[i],
                f"mfcc_{i + 1}"
            )
        )

        features.update(
            calculate_statistics(
                delta[i],
                f"delta_mfcc_{i + 1}"
            )
        )

        features.update(
            calculate_statistics(
                delta2[i],
                f"delta_delta_mfcc_{i + 1}"
            )
        )

    # ========================================================
    # MEL SPECTROGRAM
    # ========================================================

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_mean = np.mean(
        mel_db,
        axis=1
    )

    mel_std = np.std(
        mel_db,
        axis=1
    )

    for i in range(N_MELS):

        features[
            f"mel_band_{i + 1}_mean"
        ] = float(
            mel_mean[i]
        )

        features[
            f"mel_band_{i + 1}_std"
        ] = float(
            mel_std[i]
        )

    # ========================================================
    # SPECTRAL FEATURES
    # ========================================================

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    features.update(
        calculate_statistics(
            spectral_centroid,
            "spectral_centroid"
        )
    )

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    features.update(
        calculate_statistics(
            spectral_bandwidth,
            "spectral_bandwidth"
        )
    )

    spectral_rolloff = (
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    features.update(
        calculate_statistics(
            spectral_rolloff,
            "spectral_rolloff"
        )
    )

    zero_crossing_rate = (
        librosa.feature.zero_crossing_rate(
            y,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    features.update(
        calculate_statistics(
            zero_crossing_rate,
            "zero_crossing_rate"
        )
    )

    rms_energy = (
        librosa.feature.rms(
            y=y,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    features.update(
        calculate_statistics(
            rms_energy,
            "rms_energy"
        )
    )

    # ========================================================
    # PITCH / F0
    # ========================================================

    try:

        f0 = librosa.yin(
            y,
            fmin=65,
            fmax=500,
            sr=sr,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH
        )

        f0 = f0[
            np.isfinite(f0)
            & (f0 > 0)
        ]

        features.update(
            calculate_statistics(
                f0,
                "pitch_f0"
            )
        )

    except Exception:

        features.update({

            "pitch_f0_mean":
                np.nan,

            "pitch_f0_std":
                np.nan,

            "pitch_f0_min":
                np.nan,

            "pitch_f0_max":
                np.nan

        })

    return features


# ============================================================
# STEP 3B
# ============================================================

def run_step_3b():

    print(
        "\n"
        + "=" * 70
        + "\nSTEP 3B - AUDIO FEATURE EXTRACTION\n"
        + "=" * 70
    )

    if not AUDIO_METADATA_PATH.exists():

        print(
            f"❌ Không tìm thấy:\n"
            f"{AUDIO_METADATA_PATH}"
        )

        return False

    df = pd.read_csv(
        AUDIO_METADATA_PATH
    )

    path_col = (
        "processed_audio_path"
        if "processed_audio_path" in df.columns
        else "audio_path"
    )

    feature_records = []

    success_count = 0
    error_count = 0

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting Features"
    ):

        record = row.to_dict()

        status_3a = str(
            row.get(
                "audio_status",
                "success"
            )
        ).lower()

        raw_path = str(
            row.get(
                path_col,
                ""
            )
        ).strip()

        audio_file = Path(
            raw_path
        )

        if not audio_file.is_absolute():

            audio_file = (
                PROJECT_ROOT /
                audio_file
            )

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        if (
            status_3a == "success"
            and raw_path
            and audio_file.exists()
        ):

            try:

                feats = extract_all_features(
                    audio_file
                )

                record.update(feats)

                record["feature_status"] = (
                    "success"
                )

                record["feature_error"] = ""

                success_count += 1

            except Exception as e:

                record["feature_status"] = (
                    "error"
                )

                record["feature_error"] = str(e)

                error_count += 1

        else:

            record["feature_status"] = (
                "error"
            )

            record["feature_error"] = (
                f"File không tồn tại hoặc "
                f"Step 3A lỗi: {audio_file}"
            )

            error_count += 1

        feature_records.append(record)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out_df = pd.DataFrame(
        feature_records
    )

    out_df.to_csv(
        AUDIO_FEATURES_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "✅ STEP 3B HOÀN THÀNH"
    )

    print(
        f"   Thành công : {success_count}"
    )

    print(
        f"   Lỗi        : {error_count}"
    )

    print(
        f"   Output     : {AUDIO_FEATURES_PATH}"
    )

    print(
        "=" * 70
    )

    return success_count > 0


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    success_3a = run_step_3a()

    if success_3a:

        run_step_3b()

    else:

        print(
            "\n❌ STEP 3A không tạo được audio segment."
        )

        print(
            "   STEP 3B sẽ không được chạy."
        )
