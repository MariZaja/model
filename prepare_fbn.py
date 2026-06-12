# """
# Przygotowanie danych do modelu FBN.
# Wejście:  e01_audio_prep.csv, e01_video_prep.csv, e01_eeg_prep.csv
# Wyjście:  dane/audio.csv, dane/video.csv, dane/eeg.csv
#           każdy plik: kolumny identyfikacyjne + PC1..PC20
#           PC1-PC3:  komponenty LDA (max dyskryminowalność emocji)
#           PC4-PC20: komponenty PCA (wariancja rezydualna)
# """

# import os
# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler

# N_LDA = 3          # n_classes - 1 = 4 - 1
# N_PCA = 17         # reszta do 20
# N_COMPONENTS = N_LDA + N_PCA

# os.makedirs("dane", exist_ok=True)

# ID_COLS = ["trial_id", "window_id", "window_start_s", "window_end_s"]
# EMOTION_MAP = {"Anger": 0, "Sadness": 1, "Happiness": 2, "Calm": 3}


# def load_labels() -> pd.DataFrame:
#     dq = pd.read_csv("prep/e03_data_quality_prep.csv")
#     dq[["trial_id", "window_id"]] = (
#         dq["window_id"].str.split("_", expand=True).astype(int)
#     )
#     dq["window_id"] += 1
#     dq["emotion_int"] = dq["window_emotion"].map(EMOTION_MAP)
#     return dq[["trial_id", "window_id", "emotion_int"]]


# def apply_lda_pca(df, labels_df, id_cols, sentinel=None):
#     feat_cols = [c for c in df.columns if c not in id_cols]

#     merged = df.merge(labels_df, on=["trial_id", "window_id"], how="inner")
#     y = merged["emotion_int"].values
#     X = merged[feat_cols].values.astype(float)

#     if sentinel is not None:
#         X[X == sentinel] = np.nan
#     if np.isnan(X).any():
#         X = SimpleImputer(strategy="median").fit_transform(X)

#     X = StandardScaler().fit_transform(X)

#     lda = LinearDiscriminantAnalysis(n_components=N_LDA)
#     X_lda = lda.fit_transform(X, y)

#     pca = PCA(n_components=N_PCA, random_state=42)
#     X_pca = pca.fit_transform(X)

#     X_out = np.hstack([X_lda, X_pca])

#     out = merged[id_cols].copy().reset_index(drop=True)
#     for i in range(N_COMPONENTS):
#         out[f"PC{i + 1}"] = X_out[:, i]

#     return out, pca.explained_variance_ratio_.sum()


# labels = load_labels()

# # ── Audio ─────────────────────────────────────────────────────────────────────
# df_audio = pd.read_csv("prep/e03_audio_prep.csv")
# audio_out, audio_var = apply_lda_pca(df_audio, labels, ID_COLS, sentinel=-201)
# audio_out.to_csv("dane/audio3.csv", index=False)
# print(f"Audio → dane/audio3.csv   {audio_out.shape}  PCA variance: {audio_var:.3f}")

# # ── Video ─────────────────────────────────────────────────────────────────────
# df_video = pd.read_csv("prep/e03_video_prep.csv")
# video_out, video_var = apply_lda_pca(df_video, labels, ID_COLS)
# video_out.to_csv("dane/video3.csv", index=False)
# print(f"Video → dane/video3.csv   {video_out.shape}  PCA variance: {video_var:.3f}")

# # ── EEG ───────────────────────────────────────────────────────────────────────
# df_eeg = pd.read_csv("prep/e03_eeg_prep.csv")

# EEG_ID = {
#     "('trial_id', '')":       "trial_id",
#     "('window_id', '')":      "window_id",
#     "('window_start_s', '')": "window_start_s",
#     "('window_end_s', '')":   "window_end_s",
# }
# df_eeg = df_eeg.rename(columns=EEG_ID)
# df_eeg["trial_id"]  += 1   # EEG jest 0-indexed; labels i pozostałe modality są 1-indexed
# df_eeg["window_id"] += 1
# eeg_out, eeg_var = apply_lda_pca(df_eeg, labels, ID_COLS)
# eeg_out.to_csv("dane/eeg3.csv", index=False)
# print(f"EEG   → dane/eeg3.csv     {eeg_out.shape}  PCA variance: {eeg_var:.3f}")


import pandas as pd

def fill_audio_bad(path="prep/e01_data_quality_prep.csv"):
    df = pd.read_csv(path)
    df["audio_flag"] = df["audio_flag"].fillna("BAD")
    df.to_csv(path, index=False)

fill_audio_bad()
