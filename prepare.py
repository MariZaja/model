import pandas as pd

# --- surowe/e03_audio ---
df_audio = pd.read_csv("surowe/e03_audio.csv")

df_audio = df_audio[df_audio["window_id"] <= 66]
df_audio = df_audio.drop(columns=["file_id", "entity_id"])

df_audio.to_csv("surowe/e03_audio_prep.csv", index=False)
print(f"surowe/e03_audio: {df_audio.shape[0]} rows, {df_audio.shape[1]} columns")
print(f"windows per trial: {df_audio.groupby('trial_id')['window_id'].count().unique()}")

# --- surowe/e03_video ---
df_video = pd.read_csv("surowe/e03_video.csv")

df_video = df_video[df_video["window_id"] <= 66]
df_video = df_video.drop(columns=["file_id", "entity_id"])

df_video.to_csv("surowe/e03_video_prep.csv", index=False)
print(f"surowe/e03_video: {df_video.shape[0]} rows, {df_video.shape[1]} columns")
print(f"windows per trial: {df_video.groupby('trial_id')['window_id'].count().unique()}")

# --- surowe/e03_eeg ---
# column names are tuple-strings due to multi-level index export
df_eeg = pd.read_csv("surowe/e03_eeg.csv")

COL_TRIAL  = "('trial_id', '')"
COL_WINDOW = "('window_id', '')"
COL_ENTITY = "('entity_id', '')"

# window_id is 0-based (0–66); keep 0–65 for 66 windows
df_eeg = df_eeg[df_eeg[COL_WINDOW] <= 65]
df_eeg = df_eeg.drop(columns=[COL_ENTITY])

df_eeg.to_csv("surowe/e03_eeg_prep.csv", index=False)
print(f"surowe/e03_eeg:   {df_eeg.shape[0]} rows, {df_eeg.shape[1]} columns")
print(f"windows per trial: {df_eeg.groupby(COL_TRIAL)[COL_WINDOW].count().unique()}")

# --- surowe/e03_data_quality ---
df_dq = pd.read_csv("surowe/e03_data_quality.csv")

# window_id format: "{trial}_{window}", window is 0-based; keep 0–65 for 66 windows
df_dq["_win"] = df_dq["window_id"].str.split("_").str[-1].astype(int)
df_dq = df_dq[df_dq["_win"] <= 65].drop(columns=["_win"])

df_dq.to_csv("surowe/e03_data_quality_prep.csv", index=False)
print(f"surowe/e03_data_quality: {df_dq.shape[0]} rows, {df_dq.shape[1]} columns")
df_dq["_trial"] = df_dq["window_id"].str.split("_").str[0]
print(f"windows per trial: {df_dq.groupby('_trial')['window_id'].count().unique()}")
