
import os
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import librosa
import soundfile as sf

from tqdm import tqdm


# Tắt các warning không cần thiết
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(
    __file__
).resolve().parent


# ============================================================
# 2. INPUT PATH
# ============================================================

METADATA_PATH = (
    PROJECT_ROOT
    / "processed"
    / "metadata.csv"
)


# ============================================================
# 3. OUTPUT PATH
# ============================================================

# Thư mục Step 3

STEP3_DIR = (
    PROJECT_ROOT
    / "processed"
    / "step3"
)


# Thư mục chứa các audio segment

AUDIO_SEGMENTS_DIR = (
    STEP3_DIR
    / "audio_segments"
)


# File metadata sau khi xử lý audio

AUDIO_METADATA_PATH = (
    STEP3_DIR
    / "audio_metadata.csv"
)


# ============================================================
# 4. AUDIO SETTINGS
# ============================================================

# Sampling rate chuẩn

TARGET_SR = 16000


# Audio mono

MONO = True


# Có normalize amplitude hay không

NORMALIZE_AUDIO = True


# ============================================================
# 5. CREATE OUTPUT DIRECTORIES
# ============================================================

STEP3_DIR.mkdir(
    parents=True,
    exist_ok=True
)


AUDIO_SEGMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 6. FIND COLUMN
# ============================================================

def find_column(
    df,
    possible_names
):
    """
    Tự động tìm tên cột trong metadata.csv.

    Ví dụ các tên sau đều có thể được nhận diện:

    participant_id
    Participant_ID
    Participant ID

    segment_id
    Segment_ID
    Segment ID
    """

    # Dictionary lưu:
    #
    # tên đã chuẩn hóa -> tên cột thực tế

    normalized_columns = {}


    # Duyệt tất cả cột trong DataFrame

    for column in df.columns:

        # Chuẩn hóa tên cột

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        normalized_columns[
            normalized
        ] = column


    # Kiểm tra từng tên có thể

    for name in possible_names:

        normalized_name = (
            name
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )


        # Nếu tìm thấy

        if (
            normalized_name
            in normalized_columns
        ):

            return normalized_columns[
                normalized_name
            ]


    # Không tìm thấy

    return None


# ============================================================
# 7. RESOLVE AUDIO PATH
# ============================================================

def resolve_audio_path(
    audio_path
):
    """
    Chuyển audio_path thành Path.

    Hỗ trợ:

    1. Absolute path

    Ví dụ:
    /Applications/NCKH_DATATEAM/dataset/audio.wav

    2. Relative path

    Ví dụ:
    dataset/audio.wav

    3. processed/... path
    """

    # --------------------------------------------------------
    # Kiểm tra None / NaN
    # --------------------------------------------------------

    if (
        audio_path is None
        or pd.isna(audio_path)
    ):

        return None


    # --------------------------------------------------------
    # Chuyển thành string
    # --------------------------------------------------------

    audio_path = str(
        audio_path
    ).strip()


    # --------------------------------------------------------
    # Kiểm tra đường dẫn rỗng
    # --------------------------------------------------------

    if not audio_path:

        return None


    # --------------------------------------------------------
    # Các giá trị không có audio
    # --------------------------------------------------------

    unavailable_values = [

        "not_available",

        "not available",

        "none",

        "nan",

        "null",

        "missing",

        "na"

    ]


    if (
        audio_path.lower()
        in unavailable_values
    ):

        return None


    # --------------------------------------------------------
    # Tạo Path
    # --------------------------------------------------------

    path = Path(
        audio_path
    )


    # --------------------------------------------------------
    # Trường hợp 1:
    # Absolute path
    # --------------------------------------------------------

    if path.is_absolute():

        return path


    # --------------------------------------------------------
    # Trường hợp 2:
    # Relative path từ PROJECT_ROOT
    # --------------------------------------------------------

    candidate_1 = (
        PROJECT_ROOT
        / path
    )


    if candidate_1.exists():

        return candidate_1


    # --------------------------------------------------------
    # Trường hợp 3:
    # Relative path từ processed
    # --------------------------------------------------------

    candidate_2 = (
        PROJECT_ROOT
        / "processed"
        / path
    )


    if candidate_2.exists():

        return candidate_2


    # --------------------------------------------------------
    # Trường hợp 4:
    # Trả về candidate_1
    #
    # để lỗi hiển thị rõ đường dẫn
    # --------------------------------------------------------

    return candidate_1


# ============================================================
# 8. LOAD AUDIO SEGMENT
# ============================================================

def load_audio_segment(
    audio_path,
    start_time=None,
    end_time=None
):
    """
    Load audio.

    Nếu có:
        start_time
        end_time

    thì chỉ lấy đúng segment.

    Nếu không có:
        đọc toàn bộ audio.

    Returns:
        y
        sr
        resolved_audio_path
    """


    # --------------------------------------------------------
    # Resolve path
    # --------------------------------------------------------

    audio_file = resolve_audio_path(
        audio_path
    )


    # --------------------------------------------------------
    # Kiểm tra path
    # --------------------------------------------------------

    if audio_file is None:

        raise FileNotFoundError(
            "Audio path is empty"
        )


    # --------------------------------------------------------
    # Kiểm tra file
    # --------------------------------------------------------

    if not audio_file.exists():

        raise FileNotFoundError(
            f"Audio file not found: "
            f"{audio_file}"
        )


    # --------------------------------------------------------
    # Khởi tạo timestamp
    # --------------------------------------------------------

    start = None

    duration = None


    # ========================================================
    # START TIME
    # ========================================================

    if (
        start_time is not None
        and not pd.isna(start_time)
    ):

        start = float(
            start_time
        )


        # Không cho start âm

        if start < 0:

            start = 0.0


    # ========================================================
    # END TIME
    # ========================================================

    if (
        end_time is not None
        and not pd.isna(end_time)
    ):

        end = float(
            end_time
        )


        # Nếu có start
        # thì tính duration

        if start is not None:

            duration = (
                end
                - start
            )


            # Timestamp không hợp lệ

            if duration <= 0:

                raise ValueError(

                    f"Invalid timestamp: "
                    f"{start} -> {end}"

                )


    # ========================================================
    # LOAD AUDIO
    # ========================================================

    y, sr = librosa.load(

        str(audio_file),

        sr=TARGET_SR,

        mono=MONO,

        offset=start,

        duration=duration

    )


    # ========================================================
    # CHECK EMPTY AUDIO
    # ========================================================

    if len(y) == 0:

        raise ValueError(

            "Audio segment is empty"

        )


    # ========================================================
    # RETURN
    # ========================================================

    return (

        y,

        sr,

        audio_file

    )


# ============================================================
# 9. NORMALIZE AUDIO
# ============================================================

def normalize_audio(
    y
):
    """
    Normalize amplitude audio.

    Đưa biên độ về khoảng [-1, 1].
    """


    # Tìm amplitude lớn nhất

    max_amplitude = np.max(

        np.abs(y)

    )


    # Nếu audio im lặng hoàn toàn

    if max_amplitude == 0:

        return y


    # Normalize

    return (

        y

        / max_amplitude

    )


# ============================================================
# 10. PROCESS ONE SEGMENT
# ============================================================

def process_one_segment(

    row,

    participant_column,

    segment_column,

    audio_column,

    start_column,

    end_column

):

    """
    Xử lý một dòng trong metadata.csv.
    """


    # ========================================================
    # PARTICIPANT ID
    # ========================================================

    if participant_column:

        participant_id = str(

            row[
                participant_column
            ]

        )

    else:

        participant_id = (

            "unknown_participant"

        )


    # ========================================================
    # SEGMENT ID
    # ========================================================

    if segment_column:

        segment_id = str(

            row[
                segment_column
            ]

        )

    else:

        segment_id = (

            "unknown_segment"

        )


    # ========================================================
    # AUDIO PATH
    # ========================================================

    audio_path = row[

        audio_column

    ]


    # ========================================================
    # START TIME
    # ========================================================

    start_time = None


    if start_column:

        start_time = row[

            start_column

        ]


    # ========================================================
    # END TIME
    # ========================================================

    end_time = None


    if end_column:

        end_time = row[

            end_column

        ]


    # ========================================================
    # OUTPUT FILENAME
    # ========================================================

    # Làm sạch tên file

    participant_id_safe = (

        participant_id

        .replace(

            "/",

            "_"

        )

        .replace(

            "\\",

            "_"

        )

    )


    segment_id_safe = (

        segment_id

        .replace(

            "/",

            "_"

        )

        .replace(

            "\\",

            "_"

        )

    )


    # Tên file cuối cùng

    output_filename = (

        f"{participant_id_safe}"

        f"_{segment_id_safe}"

        f".wav"

    )


    # Đường dẫn output

    output_path = (

        AUDIO_SEGMENTS_DIR

        / output_filename

    )


    # ========================================================
    # LOAD AUDIO
    # ========================================================

    (

        y,

        sr,

        resolved_audio_path

    ) = load_audio_segment(

        audio_path,

        start_time,

        end_time

    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    if NORMALIZE_AUDIO:

        y = normalize_audio(

            y

        )


    # ========================================================
    # SAVE AUDIO
    # ========================================================

    sf.write(

        str(output_path),

        y,

        sr

    )


    # ========================================================
    # CALCULATE DURATION
    # ========================================================

    audio_duration = (

        len(y)

        / sr

    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "processed_audio_path":

            str(

                output_path

            ),


        "resolved_original_audio_path":

            str(

                resolved_audio_path

            ),


        "audio_duration":

            float(

                audio_duration

            ),


        "audio_sampling_rate":

            int(

                sr

            ),


        "audio_status":

            "success",


        "audio_error":

            ""

    }


# ============================================================
# 11. MAIN
# ============================================================

def main():


    # ========================================================
    # HEADER
    # ========================================================

    print()

    print(

        "=" * 70

    )

    print(

        "STEP 3A - AUDIO SEGMENTATION"

    )

    print(

        "=" * 70

    )


    # ========================================================
    # CHECK METADATA PATH
    # ========================================================

    print()

    print(

        "Metadata path:"

    )

    print(

        METADATA_PATH

    )


    # ========================================================
    # CHECK METADATA FILE
    # ========================================================

    if not METADATA_PATH.exists():


        print()

        print(

            "❌ KHÔNG TÌM THẤY metadata.csv"

        )


        print()

        print(

            "Code đang tìm file tại:"

        )


        print(

            METADATA_PATH

        )


        print()

        print(

            "Hãy kiểm tra thư mục:"

        )


        print(

            PROJECT_ROOT

            / "processed"

        )


        return


    # ========================================================
    # LOAD METADATA
    # ========================================================

    print()

    print(

        "Đang đọc metadata.csv..."

    )


    df = pd.read_csv(

        METADATA_PATH

    )


    print()

    print(

        f"Đã đọc "

        f"{len(df)} rows."

    )


    # ========================================================
    # PRINT COLUMNS
    # ========================================================

    print()

    print(

        "Các cột trong metadata.csv:"

    )


    for column in df.columns:

        print(

            f"  - {column}"

        )


    # ========================================================
    # DETECT PARTICIPANT COLUMN
    # ========================================================

    participant_column = find_column(

        df,

        [

            "participant_id",

            "participant",

            "participantid",

            "participant_ID",

            "Participant_ID"

        ]

    )


    # ========================================================
    # DETECT SEGMENT COLUMN
    # ========================================================

    segment_column = find_column(

        df,

        [

            "segment_id",

            "segment",

            "segmentid",

            "segment_ID",

            "Segment_ID"

        ]

    )


    # ========================================================
    # DETECT AUDIO COLUMN
    # ========================================================

    audio_column = find_column(

        df,

        [

            "audio_path",

            "audio",

            "audio_file",

            "audio_filepath",

            "audio_pathname",

            "Audio_Path",

            "Audio"

        ]

    )


    # ========================================================
    # DETECT START COLUMN
    # ========================================================

    start_column = find_column(

        df,

        [

            "start_time",

            "start",

            "starttime",

            "Start_Time",

            "Start"

        ]

    )


    # ========================================================
    # DETECT END COLUMN
    # ========================================================

    end_column = find_column(

        df,

        [

            "end_time",

            "end",

            "endtime",

            "End_Time",

            "End"

        ]

    )


    # ========================================================
    # PRINT DETECTED COLUMNS
    # ========================================================

    print()

    print(

        "=" * 70

    )

    print(

        "CÁC CỘT ĐƯỢC PHÁT HIỆN"

    )

    print(

        "=" * 70

    )


    print()

    print(

        f"Participant column: "

        f"{participant_column}"

    )


    print(

        f"Segment column: "

        f"{segment_column}"

    )


    print(

        f"Audio column: "

        f"{audio_column}"

    )


    print(

        f"Start time column: "

        f"{start_column}"

    )


    print(

        f"End time column: "

        f"{end_column}"

    )


    # ========================================================
    # CHECK AUDIO COLUMN
    # ========================================================

    if audio_column is None:


        print()

        print(

            "❌ LỖI: "

            "Không tìm thấy cột audio_path."

        )


        print()

        print(

            "Các cột hiện có là:"

        )


        print(

            list(

                df.columns

            )

        )


        return


    # ========================================================
    # CHECK TIMESTAMP
    # ========================================================

    if (

        start_column is not None

        and end_column is not None

    ):


        print()

        print(

            "✅ Tìm thấy Start Time "

            "và End Time."

        )


        print(

            "Audio sẽ được cắt "

            "theo timestamp."

        )
    else:


        print()

        print(

            "⚠️ Không tìm thấy đầy đủ "

            "Start Time / End Time."

        )


        print(

            "Code sẽ xử lý toàn bộ audio."

        )


    # ========================================================
    # PROCESS AUDIO
    # ========================================================

    results = []


    print()

    print(

        "=" * 70

    )

    print(

        "BẮT ĐẦU XỬ LÝ AUDIO"

    )

    print(

        "=" * 70

    )


    for idx, row in tqdm(

        df.iterrows(),

        total=len(df),

        desc="Processing Audio"

    ):


        # ----------------------------------------------------
        # COPY ORIGINAL METADATA
        # ----------------------------------------------------

        record = row.to_dict()


        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        try:


            audio_result = (

                process_one_segment(

                    row,

                    participant_column,

                    segment_column,

                    audio_column,

                    start_column,

                    end_column

                )

            )


            # ------------------------------------------------
            # ADD AUDIO RESULT
            # ------------------------------------------------

            record.update(

                audio_result

            )


        except Exception as e:


            # ------------------------------------------------
            # ERROR
            # ------------------------------------------------

            record.update({

                "processed_audio_path":

                    "",


                "resolved_original_audio_path":

                    "",


                "audio_duration":

                    None,


                "audio_sampling_rate":

                    None,


                "audio_status":

                    "error",


                "audio_error":

                    str(e)

            })


        # ----------------------------------------------------
        # APPEND
        # ----------------------------------------------------

        results.append(

            record

        )


    # ========================================================
    # CREATE OUTPUT DATAFRAME
    # ========================================================

    output_df = pd.DataFrame(

        results

    )


    # ========================================================
    # ENSURE OUTPUT DIRECTORIES EXIST
    # ========================================================

    # Đây là lớp bảo vệ thứ hai.
    #
    # Ngay cả khi thư mục bị xóa giữa quá trình chạy,
    # code vẫn tạo lại trước khi lưu CSV.

    STEP3_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    AUDIO_SEGMENTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # ========================================================
    # SAVE AUDIO METADATA
    # ========================================================

    print()

    print(

        "Đang lưu audio_metadata.csv..."

    )


    output_df.to_csv(

        AUDIO_METADATA_PATH,

        index=False,

        encoding="utf-8-sig"

    )


    # ========================================================
    # STATISTICS
    # ========================================================

    success_count = len(

        output_df[

            output_df[

                "audio_status"

            ]

            == "success"

        ]

    )


    error_count = len(

        output_df[

            output_df[

                "audio_status"

            ]

            == "error"

        ]

    )


    # ========================================================
    # COUNT AUDIO FILES
    # ========================================================

    audio_file_count = len(

        list(

            AUDIO_SEGMENTS_DIR.glob(

                "*.wav"

            )

        )

    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()

    print(

        "=" * 70

    )

    print(

        "STEP 3A COMPLETED"

    )

    print(

        "=" * 70

    )


    print()

    print(

        f"Tổng segments: "

        f"{len(output_df)}"

    )


    print(

        f"Thành công: "

        f"{success_count}"

    )


    print(

        f"Lỗi: "

        f"{error_count}"

    )


    print(

        f"Số file WAV đã tạo: "

        f"{audio_file_count}"

    )


    print()

    print(

        "Audio segments được lưu tại:"

    )


    print(

        AUDIO_SEGMENTS_DIR

    )


    print()

    print(

        "Metadata được lưu tại:"

    )


    print(

        AUDIO_METADATA_PATH

    )


    print()

    print(

        "=" * 70

    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()

# ============================================================
# STEP 3B - AUDIO FEATURE EXTRACTION
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import warnings

from pathlib import Path

import numpy as np
import pandas as pd

import librosa

from tqdm import tqdm


# Tắt các warning không cần thiết trong quá trình chạy

warnings.filterwarnings(
    "ignore"
)


# ============================================================
# 2. PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


# ============================================================
# 3. INPUT PATH
# ============================================================
AUDIO_METADATA_PATH = (

    PROJECT_ROOT

    / "processed"

    / "step3"

    / "audio_metadata.csv"

)


# ============================================================
# 4. AUDIO SEGMENTS DIRECTORY
# ============================================================
AUDIO_SEGMENTS_DIR = (

    PROJECT_ROOT

    / "processed"

    / "step3"

    / "audio_segments"

)


# ============================================================
# 5. OUTPUT PATH
# ============================================================
AUDIO_FEATURES_PATH = (

    PROJECT_ROOT

    / "processed"

    / "step3"

    / "audio_features.csv"

)


# ============================================================
# 6. AUDIO PARAMETERS
# ============================================================
TARGET_SR = 16000

N_MFCC = 13

N_MELS = 128

N_FFT = 2048

HOP_LENGTH = 512


# ============================================================
# 7. CALCULATE STATISTICS
# ============================================================

def calculate_statistics(
    values,
    prefix
):
    """
    Tính các thống kê cơ bản cho một vector đặc trưng.

    Các thống kê được sử dụng:

        mean
        std
        min
        max

    Parameters
    ----------
    values:
        Vector hoặc mảng đặc trưng.

    prefix:
        Tiền tố dùng để đặt tên cho các cột kết quả.

    Returns
    -------
    Dictionary chứa các giá trị thống kê.
    """

    # --------------------------------------------------------
    # Chuyển dữ liệu thành numpy array
    # --------------------------------------------------------

    values = np.asarray(

        values,

        dtype=np.float64

    )


    # --------------------------------------------------------
    # Loại bỏ các giá trị NaN và Inf
    # --------------------------------------------------------

    values = values[

        np.isfinite(

            values

        )

    ]
    # --------------------------------------------------------
    # Kiểm tra trường hợp không có dữ liệu hợp lệ
    # --------------------------------------------------------

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
    # --------------------------------------------------------
    # Tính các thống kê
    # --------------------------------------------------------

    return {

        f"{prefix}_mean":

            float(

                np.mean(

                    values

                )

            ),

        f"{prefix}_std":

            float(

                np.std(

                    values

                )

            ),

        f"{prefix}_min":

            float(

                np.min(

                    values

                )

            ),

        f"{prefix}_max":

            float(

                np.max(

                    values

                )

            )

    }
# ============================================================
# 8. EXTRACT MFCC FEATURES
# ============================================================

def extract_mfcc_features(
    y,
    sr
):
    """
    Trích xuất các đặc trưng:

    1. MFCC
    2. Delta MFCC
    3. Delta-Delta MFCC

    MFCC thường được sử dụng để mô tả đặc điểm
    phổ âm thanh và đặc trưng giọng nói.
    """

    # --------------------------------------------------------
    # Tính MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(

        y=y,

        sr=sr,

        n_mfcc=N_MFCC,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH

    )
    # --------------------------------------------------------
    # Tính Delta MFCC
    # --------------------------------------------------------

    delta = librosa.feature.delta(

        mfcc

    )


    # --------------------------------------------------------
    # Tính Delta-Delta MFCC
    # --------------------------------------------------------

    delta_delta = librosa.feature.delta(

        mfcc,

        order=2

    )


    # --------------------------------------------------------
    # Khởi tạo dictionary lưu kết quả
    # --------------------------------------------------------

    features = {}


    # ========================================================
    # MFCC
    # ========================================================

    for i in range(

        N_MFCC

    ):

        # Tính thống kê cho từng MFCC

        stats = calculate_statistics(

            mfcc[i],

            f"mfcc_{i + 1}"

        )


        # Thêm vào dictionary

        features.update(

            stats

        )


    # ========================================================
    # DELTA MFCC
    # ========================================================

    for i in range(

        N_MFCC

    ):

        # Tính thống kê cho Delta MFCC

        stats = calculate_statistics(

            delta[i],

            f"delta_mfcc_{i + 1}"

        )


        # Thêm vào dictionary

        features.update(

            stats

        )


    # ========================================================
    # DELTA-DELTA MFCC
    # ========================================================

    for i in range(

        N_MFCC

    ):

        # Tính thống kê cho Delta-Delta MFCC

        stats = calculate_statistics(

            delta_delta[i],

            f"delta_delta_mfcc_{i + 1}"

        )


        # Thêm vào dictionary

        features.update(

            stats

        )


    # --------------------------------------------------------
    # Trả về toàn bộ đặc trưng MFCC
    # --------------------------------------------------------

    return features


# ============================================================
# 9. EXTRACT MEL-SPECTROGRAM FEATURES
# ============================================================

def extract_mel_features(
    y,
    sr
):
    """
    Trích xuất đặc trưng Mel-Spectrogram.

    Mel-Spectrogram được sử dụng để biểu diễn
    năng lượng của tín hiệu âm thanh trên các
    dải tần Mel.

    Có 128 Mel bands.

    Để mỗi segment chỉ chiếm một dòng trong dataset,
    chúng ta lưu giá trị mean và std của từng Mel band.
    """

    # --------------------------------------------------------
    # Tính Mel-Spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(

        y=y,

        sr=sr,

        n_mels=N_MELS,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH

    )


    # --------------------------------------------------------
    # Chuyển đổi sang đơn vị decibel
    # --------------------------------------------------------

    mel_db = librosa.power_to_db(

        mel,

        ref=np.max

    )


    # --------------------------------------------------------
    # Tính mean theo chiều thời gian
    # --------------------------------------------------------

    mel_mean = np.mean(

        mel_db,

        axis=1

    )


    # --------------------------------------------------------
    # Tính standard deviation theo chiều thời gian
    # --------------------------------------------------------

    mel_std = np.std(

        mel_db,

        axis=1

    )


    # --------------------------------------------------------
    # Khởi tạo dictionary
    # --------------------------------------------------------

    features = {}


    # ========================================================
    # LƯU MEAN CỦA TỪNG MEL BAND
    # ========================================================

    for i in range(

        N_MELS

    ):

        features[

            f"mel_band_{i + 1}_mean"

        ] = float(

            mel_mean[i]

        )


    # ========================================================
    # LƯU STD CỦA TỪNG MEL BAND
    # ========================================================

    for i in range(

        N_MELS

    ):

        features[

            f"mel_band_{i + 1}_std"

        ] = float(

            mel_std[i]

        )


    # --------------------------------------------------------
    # Trả về đặc trưng Mel-Spectrogram
    # --------------------------------------------------------

    return features


# ============================================================
# 10. EXTRACT SPECTRAL FEATURES
# ============================================================

def extract_spectral_features(
    y,
    sr
):
    """
    Trích xuất các đặc trưng phổ âm thanh:

    1. Spectral Centroid
    2. Spectral Bandwidth
    3. Spectral Rolloff
    4. Zero Crossing Rate
    5. RMS Energy
    6. Chroma
    """

    # ========================================================
    # SPECTRAL CENTROID
    # ========================================================

    spectral_centroid = (

        librosa.feature.spectral_centroid(

            y=y,

            sr=sr,

            n_fft=N_FFT,

            hop_length=HOP_LENGTH

        )

    )


    # ========================================================
    # SPECTRAL BANDWIDTH
    # ========================================================

    spectral_bandwidth = (

        librosa.feature.spectral_bandwidth(

            y=y,

            sr=sr,

            n_fft=N_FFT,

            hop_length=HOP_LENGTH

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


    # ========================================================
    # ZERO CROSSING RATE
    # ========================================================

    zero_crossing_rate = (

        librosa.feature.zero_crossing_rate(

            y,

            frame_length=N_FFT,

            hop_length=HOP_LENGTH

        )

    )


    # ========================================================
    # RMS ENERGY
    # ========================================================

    rms = (

        librosa.feature.rms(

            y=y,

            frame_length=N_FFT,

            hop_length=HOP_LENGTH

        )

    )


    # ========================================================
    # CHROMA
    # ========================================================

    chroma = (

        librosa.feature.chroma_stft(

            y=y,

            sr=sr,

            n_fft=N_FFT,

            hop_length=HOP_LENGTH

        )

    )


    # ========================================================
    # KHỞI TẠO DICTIONARY
    # ========================================================

    features = {}


    # ========================================================
    # SPECTRAL CENTROID STATISTICS
    # ========================================================

    features.update(

        calculate_statistics(

            spectral_centroid,

            "spectral_centroid"

        )

    )


    # ========================================================
    # SPECTRAL BANDWIDTH STATISTICS
    # ========================================================

    features.update(

        calculate_statistics(

            spectral_bandwidth,

            "spectral_bandwidth"

        )

    )


    # ========================================================
    # SPECTRAL ROLLOFF STATISTICS
    # ========================================================

    features.update(

        calculate_statistics(

            spectral_rolloff,

            "spectral_rolloff"

        )

    )


    # ========================================================
    # ZERO CROSSING RATE STATISTICS
    # ========================================================

    features.update(

        calculate_statistics(

            zero_crossing_rate,

            "zero_crossing_rate"

        )

    )


    # ========================================================
    # RMS ENERGY STATISTICS
    # ========================================================

    features.update(

        calculate_statistics(

            rms,

            "rms_energy"

        )

    )


    # ========================================================
    # CHROMA STATISTICS
    # ========================================================

    for i in range(

        chroma.shape[0]

    ):

        # Tính thống kê cho từng Chroma band

        stats = calculate_statistics(

            chroma[i],

            f"chroma_{i + 1}"

        )


        # Thêm vào dictionary

        features.update(

            stats

        )


    # --------------------------------------------------------
    # Trả về các đặc trưng spectral
    # --------------------------------------------------------

    return features


# ============================================================
# 11. EXTRACT PITCH / F0 FEATURES
# ============================================================

def extract_pitch_features(
    y,
    sr
):
    """
    Ước tính Fundamental Frequency (F0).

    F0 là tần số cơ bản của giọng nói và có thể
    phản ánh một số đặc điểm liên quan đến cao độ giọng.

    Khoảng tần số được sử dụng:

        65 Hz - 500 Hz
    """

    try:

        # ----------------------------------------------------
        # Ước tính F0 bằng thuật toán YIN
        # ----------------------------------------------------

        f0 = librosa.yin(

            y,

            fmin=65,

            fmax=500,

            sr=sr,

            frame_length=N_FFT,

            hop_length=HOP_LENGTH

        )


        # ----------------------------------------------------
        # Loại bỏ giá trị NaN và Inf
        # ----------------------------------------------------

        f0 = f0[

            np.isfinite(

                f0

            )

        ]


        # ----------------------------------------------------
        # Chỉ giữ các giá trị F0 dương
        # ----------------------------------------------------

        f0 = f0[

            f0 > 0

        ]


        # ----------------------------------------------------
        # Tính thống kê F0
        # ----------------------------------------------------

        return calculate_statistics(

            f0,

            "pitch_f0"

        )


    except Exception:

        # ----------------------------------------------------
        # Nếu không thể ước tính F0
        # ----------------------------------------------------

        return {

            "pitch_f0_mean":
                np.nan,

            "pitch_f0_std":
                np.nan,

            "pitch_f0_min":
                np.nan,

            "pitch_f0_max":
                np.nan

        }


# ============================================================
# 12. EXTRACT ALL FEATURES FOR ONE AUDIO
# ============================================================

def extract_all_features(
    audio_path
):
    """
    Trích xuất toàn bộ đặc trưng âm thanh
    cho một audio segment.
    """

    y, sr = librosa.load(

        str(audio_path),

        sr=TARGET_SR,

        mono=True

    )

    if len(y) == 0:

        raise ValueError(

            "Audio is empty"

        )

    features = {}

    duration = (

        len(y)

        / sr

    )


    features[

        "feature_audio_duration"

    ] = float(

        duration

    )

    features[

        "feature_sampling_rate"

    ] = int(

        sr

    )

    mfcc_features = (

        extract_mfcc_features(

            y,

            sr

        )

    )


    features.update(

        mfcc_features

    )

    mel_features = (

        extract_mel_features(

            y,

            sr

        )

    )


    features.update(

        mel_features

    )

    spectral_features = (

        extract_spectral_features(

            y,

            sr

        )

    )


    features.update(

        spectral_features

    )

    pitch_features = (

        extract_pitch_features(

            y,

            sr

        )

    )


    features.update(

        pitch_features

    )

    return features


def main():

    # ========================================================
    # HEADER
    # ========================================================

    print()

    print(

        "=" * 70

    )

    print(

        "STEP 3B - AUDIO FEATURE EXTRACTION"

    )

    print(

        "=" * 70

    )

    print()

    print(

        "Audio metadata path:"

    )

    print(

        AUDIO_METADATA_PATH

    )


    if not AUDIO_METADATA_PATH.exists():

        print()

        print(

            "❌ KHÔNG TÌM THẤY audio_metadata.csv"

        )

        print()

        print(

            "Hãy chạy Step 3A trước."

        )

        return



    if not AUDIO_SEGMENTS_DIR.exists():

        print()

        print(

            "❌ KHÔNG TÌM THẤY THƯ MỤC audio_segments"

        )

        print()

        print(

            AUDIO_SEGMENTS_DIR

        )

        return


    print()

    print(

        "Đang đọc audio_metadata.csv..."

    )


    df = pd.read_csv(

        AUDIO_METADATA_PATH

    )


    print()

    print(

        f"Tổng số segments: "

        f"{len(df)}"

    )

    possible_path_columns = [

        "processed_audio_path",

        "audio_path"

    ]


    audio_path_column = None


    # Tìm cột audio path

    for column in possible_path_columns:

        if column in df.columns:

            audio_path_column = column

            break


    if audio_path_column is None:

        print()

        print(

            "❌ KHÔNG TÌM THẤY CỘT AUDIO PATH"

        )

        print()

        print(

            "Các cột hiện có:"

        )

        print(

            list(

                df.columns

            )

        )

        return

    print()

    print(

        f"Audio path column: "

        f"{audio_path_column}"

    )

    feature_records = []


    # Đếm số segment thành công

    success_count = 0


    # Đếm số segment lỗi

    error_count = 0

    print()

    print(

        "=" * 70

    )

    print(

        "BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG"

    )

    print(

        "=" * 70

    )

    for idx, row in tqdm(

        df.iterrows(),

        total=len(df),

        desc="Extracting Features"

    ):


        record = row.to_dict()

        audio_path = row[

            audio_path_column

        ]

        audio_path = Path(

            str(

                audio_path

            )

        )

        if not audio_path.is_absolute():

            audio_path = (

                PROJECT_ROOT

                / audio_path

            )

        try:

            features = (

                extract_all_features(

                    audio_path

                )

            )

            record.update(

                features

            )

            record[

                "feature_status"

            ] = "success"


            record[

                "feature_error"

            ] = ""


            # Tăng số lượng thành công

            success_count += 1


        except Exception as e:

            record[

                "feature_status"

            ] = "error"

            record[

                "feature_error"

            ] = str(

                e

            )
            # Tăng số lượng lỗi

            error_count += 1

        feature_records.append(

            record

        )

    output_df = pd.DataFrame(

        feature_records

    )

    AUDIO_FEATURES_PATH.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    print()

    print(

        "Đang lưu audio_features.csv..."

    )


    output_df.to_csv(

        AUDIO_FEATURES_PATH,

        index=False,

        encoding="utf-8-sig"

    )

    print()

    print(

        "=" * 70

    )

    print(

        "STEP 3B COMPLETED"

    )

    print(

        "=" * 70

    )

    print()

    print(

        f"Tổng segments: "

        f"{len(output_df)}"

    )

    print(

        f"Trích xuất thành công: "

        f"{success_count}"

    )

    print(

        f"Lỗi: "

        f"{error_count}"

    )

    print()

    print(

        f"Tổng số columns: "

        f"{len(output_df.columns)}"

    )

    print()

    print(

        "File kết quả:"

    )


    print(

        AUDIO_FEATURES_PATH

    )
    print()

    print(

        "=" * 70

    )
# ============================================================
# 14. RUN PROGRAM
# ============================================================
if __name__ == "__main__":

    main()