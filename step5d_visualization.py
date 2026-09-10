import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "/Applications/NCKH_DATATEAM"

INPUT_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "step4",
    "audio_features_final.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed",
    "step5",
    "visualizations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

exclude_columns = [
    "participant",
    "segment_id",
    "dataset",
    "text",
    "audio_path",
    "video_path",
    "processing_status",
    "processed_audio_path",
    "resolved_original_audio_path",
    "audio_status",
    "feature_status",
    "audio_sampling_rate",
    "feature_sampling_rate"
]

feature_columns = [
    c for c in df.select_dtypes(
        include=[np.number]
    ).columns
    if c not in exclude_columns
]

feature_df = df[feature_columns]

print("=" * 70)
print("STEP 5D - VISUALIZATION & EDA")
print("=" * 70)

print(f"\nTổng số segment: {len(df)}")
print(f"Tổng số audio features: {len(feature_columns)}")

means = feature_df.mean().sort_values()

plt.figure(figsize=(12, 6))
plt.hist(means, bins=30)
plt.xlabel("Feature mean")
plt.ylabel("Number of features")
plt.title("Distribution of Audio Feature Means")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "feature_mean_distribution.png"
    ),
    dpi=300
)
plt.close()

stds = feature_df.std().sort_values()

plt.figure(figsize=(12, 6))
plt.hist(stds, bins=30)
plt.xlabel("Feature standard deviation")
plt.ylabel("Number of features")
plt.title("Distribution of Audio Feature Standard Deviations")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "feature_std_distribution.png"
    ),
    dpi=300
)
plt.close()

top_std = stds.sort_values(
    ascending=False
).head(20).sort_values()

plt.figure(figsize=(10, 8))
plt.barh(
    top_std.index,
    top_std.values
)
plt.xlabel("Standard deviation")
plt.ylabel("Feature")
plt.title("Top 20 Audio Features by Standard Deviation")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "top20_feature_std.png"
    ),
    dpi=300
)
plt.close()

variances = feature_df.var().sort_values(
    ascending=False
)

top_variance = variances.head(20).sort_values()

plt.figure(figsize=(10, 8))
plt.barh(
    top_variance.index,
    top_variance.values
)
plt.xlabel("Variance")
plt.ylabel("Feature")
plt.title("Top 20 Audio Features by Variance")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "top20_feature_variance.png"
    ),
    dpi=300
)
plt.close()

sample_features = feature_columns[:12]

for feature in sample_features:
    plt.figure(figsize=(8, 5))
    plt.hist(
        feature_df[feature],
        bins=30
    )
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(
        f"Distribution of {feature}"
    )
    plt.tight_layout()

    safe_name = feature.replace(
        "/",
        "_"
    ).replace(
        " ",
        "_"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"distribution_{safe_name}.png"
        ),
        dpi=300
    )
    plt.close()

corr_features = feature_df.var().sort_values(
    ascending=False
).head(30).index

corr_matrix = feature_df[
    corr_features
].corr()

plt.figure(figsize=(14, 12))
plt.imshow(
    corr_matrix,
    aspect="auto"
)
plt.colorbar()
plt.xticks(
    range(len(corr_features)),
    corr_features,
    rotation=90,
    fontsize=6
)
plt.yticks(
    range(len(corr_features)),
    corr_features,
    fontsize=6
)
plt.title(
    "Correlation Matrix of Top 30 Variable Audio Features"
)
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_matrix_top30.png"
    ),
    dpi=300
)
plt.close()

summary = pd.DataFrame({
    "feature": feature_columns,
    "mean": feature_df.mean().values,
    "std": feature_df.std().values,
    "variance": feature_df.var().values,
    "min": feature_df.min().values,
    "median": feature_df.median().values,
    "max": feature_df.max().values
})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step5d_feature_eda_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("STEP 5D COMPLETED")
print("=" * 70)

print(f"\nSố feature phân tích: {len(feature_columns)}")

print("\nCác biểu đồ được tạo:")
print("- feature_mean_distribution.png")
print("- feature_std_distribution.png")
print("- top20_feature_std.png")
print("- top20_feature_variance.png")
print("- correlation_matrix_top30.png")
print("- 12 biểu đồ phân phối feature mẫu")

print("\nThư mục kết quả:")
print(OUTPUT_DIR)

print("\nFile thống kê EDA:")
print(
    os.path.join(
        OUTPUT_DIR,
        "step5d_feature_eda_summary.csv"
    )
)

print("\n" + "=" * 70)