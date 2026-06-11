import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro import param
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch import tensor
from torch.distributions import constraints
from scipy.stats import norm as scipy_norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Encodings ─────────────────────────────────────────────────────────────────
E_STATES = ["Angry", "Sad", "Happy", "Calm"]
Q_STATES = ["BAD", "NOISY", "GOOD"]

E_ENC = {s: i for i, s in enumerate(E_STATES)}
Q_ENC = {s: i for i, s in enumerate(Q_STATES)}
E_DEC = {i: s for s, i in E_ENC.items()}
Q_DEC = {i: s for s, i in Q_ENC.items()}

EMOTION_MAP = {"Anger": "Angry", "Sadness": "Sad", "Happiness": "Happy", "Calm": "Calm"}

V_AUDIO = [f"V{k}" for k in range(1,  21)]   # V1-V20
V_VIDEO = [f"V{k}" for k in range(21, 41)]   # V21-V40
V_EEG   = [f"V{k}" for k in range(41, 61)]   # V41-V60

MODALITIES = {
    "audio2": {"q": "Q_audio", "v_cols": V_AUDIO},
    "video2": {"q": "Q_video", "v_cols": V_VIDEO},
    "eeg2":   {"q": "Q_eeg",   "v_cols": V_EEG},
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str = "dane") -> pd.DataFrame:
    dq = pd.read_csv("prep/e02_data_quality_prep.csv")
    dq[["trial_id", "window_id"]] = (
        dq["window_id"].str.split("_", expand=True).astype(int)
    )
    dq["window_id"] += 1
    dq["E"] = dq["window_emotion"].map(EMOTION_MAP)

    base = dq[["trial_id", "window_id", "E",
               "audio_flag", "video_flag", "eeg_flag"]].copy()
    base = base.rename(columns={
        "audio_flag": "Q_audio",
        "video_flag": "Q_video",
        "eeg_flag":   "Q_eeg",
    })

    for mod, cfg in MODALITIES.items():
        feat = pd.read_csv(f"{data_dir}/{mod}.csv")
        pc_rename = {f"PC{i}": cfg["v_cols"][i - 1] for i in range(1, 21)}
        feat = feat.rename(columns=pc_rename)
        base = base.merge(
            feat[["trial_id", "window_id"] + cfg["v_cols"]],
            on=["trial_id", "window_id"],
            how="left",
        )

    base = base.dropna(subset=["E"]).reset_index(drop=True)
    return base


# ── Tensor preparation ────────────────────────────────────────────────────────

def prepare_tensors(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    tensors["E"] = torch.tensor(df["E"].map(E_ENC).values, dtype=torch.float32)
    for mod, cfg in MODALITIES.items():
        q_col = cfg["q"]
        q_vals = df[q_col].map(
            lambda x: float(Q_ENC[x]) if (pd.notna(x) and x in Q_ENC) else float("nan")
        )
        tensors[q_col] = torch.tensor(q_vals.values, dtype=torch.float32)
        for v_name in cfg["v_cols"]:
            tensors[v_name] = torch.tensor(df[v_name].values, dtype=torch.float32)
    return tensors


# ── Joint Pyro model ──────────────────────────────────────────────────────────

def joint_model(tensors: dict[str, torch.Tensor], use_quality: bool) -> None:
    N     = tensors["E"].shape[0]
    E_obs = tensors["E"].long()

    e_probs = param("E_probs", tensor([0.25, 0.25, 0.25, 0.25]),
                    constraint=constraints.simplex)

    with pyro.plate("obs", N):
        pyro.sample("E", dist.Categorical(probs=e_probs), obs=E_obs)

        for mod, cfg in MODALITIES.items():
            q_name = cfg["q"]
            q_raw  = tensors[q_name]                    # float, NaN = missing
            avail  = ~torch.isnan(q_raw)                # [N] bool mask
            safe_q = q_raw.where(avail, torch.zeros_like(q_raw)).long()

            q_probs = param(f"{q_name}_probs", tensor([1/3, 1/3, 1/3]),
                            constraint=constraints.simplex)

            if use_quality:
                with pyro.poutine.mask(mask=avail):
                    pyro.sample(q_name, dist.Categorical(probs=q_probs), obs=safe_q)

            for v_name in cfg["v_cols"]:
                v_raw  = tensors[v_name]
                v_mask = avail & ~torch.isnan(v_raw)
                safe_v = v_raw.where(v_mask, torch.zeros_like(v_raw))

                beta_0 = param(f"{v_name}_beta_0", tensor(0.0))
                beta_E = param(f"{v_name}_beta_E", tensor([0.0, 0.0, 0.0, 0.0]))
                sigma  = param(f"{v_name}_sigma",  tensor(1.0),
                               constraint=constraints.positive)

                if use_quality:
                    beta_Q = param(f"{v_name}_beta_Q", tensor([0.0, 0.0, 0.0]))
                    mu = beta_0 + beta_E[E_obs] + beta_Q[safe_q]
                else:
                    mu = beta_0 + beta_E[E_obs]

                with pyro.poutine.mask(mask=v_mask):
                    pyro.sample(v_name, dist.Normal(mu, sigma), obs=safe_v)


def joint_guide(tensors: dict[str, torch.Tensor], use_quality: bool) -> None:
    pass


# ── Joint training ────────────────────────────────────────────────────────────

def train_joint(
    train_df: pd.DataFrame,
    use_quality: bool,
    num_steps: int = 3000,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    tensors = prepare_tensors(train_df)

    svi = SVI(joint_model, joint_guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

    print(f"Joint training (SVI, {num_steps} steps, {len(train_df)} rows)…")
    for step in range(num_steps):
        loss = svi.step(tensors, use_quality)
        if step % 1000 == 0:
            print(f"  step {step:5d}  loss = {loss:.2f}")

    return {k: v.detach().clone() for k, v in pyro.get_param_store().items()}


# ── Combined inference P(E | all_V, all_Q) ───────────────────────────────────

def predict_E(
    test_df: pd.DataFrame,
    params: dict[str, torch.Tensor],
    e_prior: np.ndarray,
    use_quality: bool,
) -> pd.DataFrame:
    records = []
    for _, row in test_df.iterrows():
        log_liks = np.zeros(len(E_STATES))

        for mod, cfg in MODALITIES.items():
            q_col = cfg["q"]

            if pd.isna(row[q_col]):
                continue
            q = int(Q_ENC[row[q_col]]) if use_quality else None

            for v_name in cfg["v_cols"]:
                v_val = row[v_name]
                if pd.isna(v_val):
                    continue
                b0  = params[f"{v_name}_beta_0"].item()
                bE  = params[f"{v_name}_beta_E"].detach().numpy()
                sig = params[f"{v_name}_sigma"].item()
                if use_quality:
                    bQ     = params[f"{v_name}_beta_Q"].detach().numpy()
                    mu_arr = b0 + bE + bQ[q]
                else:
                    mu_arr = b0 + bE
                for e in range(len(E_STATES)):
                    log_liks[e] += scipy_norm.logpdf(v_val, mu_arr[e], sig)

        log_post = log_liks + np.log(e_prior + 1e-300)
        log_post -= log_post.max()
        posteriors = np.exp(log_post)
        posteriors /= posteriors.sum()

        records.append({
            "E_true": row["E"],
            "E_pred": E_DEC[int(np.argmax(posteriors))],
            **{f"P({E_STATES[e]})": f"{posteriors[e]:.3f}" for e in range(len(E_STATES))},
        })

    return pd.DataFrame(records)


# ── KL information-gain matrix  I(m,q) = E[KL(P(E|X_m,Q) ‖ P(E|Q))] ─────────

def compute_kl_matrix(
    df: pd.DataFrame,
    params: dict[str, torch.Tensor],
    e_prior: np.ndarray,
    use_quality: bool,
) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}

    for mod, cfg in MODALITIES.items():
        q_col   = cfg["q"]
        kl_by_q: dict[str, list[float]] = {q: [] for q in Q_STATES}

        for _, row in df.iterrows():
            if pd.isna(row[q_col]):
                continue
            q_state = row[q_col]
            q_idx   = int(Q_ENC[q_state]) if use_quality else None

            log_liks = np.zeros(len(E_STATES))
            for v_name in cfg["v_cols"]:
                v_val = row[v_name]
                if pd.isna(v_val):
                    continue
                b0  = params[f"{v_name}_beta_0"].item()
                bE  = params[f"{v_name}_beta_E"].detach().numpy()
                sig = params[f"{v_name}_sigma"].item()
                if use_quality:
                    bQ     = params[f"{v_name}_beta_Q"].detach().numpy()
                    mu_arr = b0 + bE + bQ[q_idx]
                else:
                    mu_arr = b0 + bE
                for e in range(len(E_STATES)):
                    log_liks[e] += scipy_norm.logpdf(v_val, mu_arr[e], sig)

            log_post = log_liks + np.log(e_prior + 1e-300)
            log_post -= log_post.max()
            posterior  = np.exp(log_post)
            posterior /= posterior.sum()

            kl = float(np.sum(
                posterior * np.log((posterior + 1e-300) / (e_prior + 1e-300))
            ))
            kl_by_q[q_state].append(kl)

        rows[mod] = {
            "Q_high": float(np.mean(kl_by_q["GOOD"]))  if kl_by_q["GOOD"]  else 0.0,
            "Q_med":  float(np.mean(kl_by_q["NOISY"])) if kl_by_q["NOISY"] else 0.0,
            "Q_low":  float(np.mean(kl_by_q["BAD"]))   if kl_by_q["BAD"]   else 0.0,
        }

    return pd.DataFrame(rows).T[["Q_high", "Q_med", "Q_low"]]


def compute_kl_matrix_per_emotion(
    df: pd.DataFrame,
    params: dict[str, torch.Tensor],
    e_prior: np.ndarray,
    use_quality: bool,
) -> dict[str, pd.DataFrame]:
    kl_store: dict[str, dict[str, dict[str, list[float]]]] = {
        e_name: {mod: {q: [] for q in Q_STATES} for mod in MODALITIES}
        for e_name in E_STATES
    }

    for _, row in df.iterrows():
        e_true = row["E"]
        if pd.isna(e_true) or e_true not in E_STATES:
            continue

        for mod, cfg in MODALITIES.items():
            q_col = cfg["q"]
            if pd.isna(row[q_col]):
                continue
            q_state = row[q_col]
            q_idx   = int(Q_ENC[q_state]) if use_quality else None

            log_liks = np.zeros(len(E_STATES))
            for v_name in cfg["v_cols"]:
                v_val = row[v_name]
                if pd.isna(v_val):
                    continue
                b0  = params[f"{v_name}_beta_0"].item()
                bE  = params[f"{v_name}_beta_E"].detach().numpy()
                sig = params[f"{v_name}_sigma"].item()
                if use_quality:
                    bQ     = params[f"{v_name}_beta_Q"].detach().numpy()
                    mu_arr = b0 + bE + bQ[q_idx]
                else:
                    mu_arr = b0 + bE
                for e in range(len(E_STATES)):
                    log_liks[e] += scipy_norm.logpdf(v_val, mu_arr[e], sig)

            log_post = log_liks + np.log(e_prior + 1e-300)
            log_post -= log_post.max()
            posterior  = np.exp(log_post)
            posterior /= posterior.sum()

            kl = float(np.sum(
                posterior * np.log((posterior + 1e-300) / (e_prior + 1e-300))
            ))
            kl_store[e_true][mod][q_state].append(kl)

    result: dict[str, pd.DataFrame] = {}
    for e_name in E_STATES:
        rows: dict[str, dict[str, float]] = {}
        for mod in MODALITIES:
            s = kl_store[e_name][mod]
            rows[mod] = {
                "Q_high": float(np.mean(s["GOOD"]))  if s["GOOD"]  else 0.0,
                "Q_med":  float(np.mean(s["NOISY"])) if s["NOISY"] else 0.0,
                "Q_low":  float(np.mean(s["BAD"]))   if s["BAD"]   else 0.0,
            }
        result[e_name] = pd.DataFrame(rows).T[["Q_high", "Q_med", "Q_low"]]

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", action="store_true",
                        help="Include quality (Q) nodes as parents of V features")
    args = parser.parse_args()
    use_quality = args.quality

    print(f"Mode: {'with quality (Q nodes active)' if use_quality else 'without quality (E only)'}")
    print("Loading data from dane/…")
    df_all = load_data()

    print(f"  Total rows: {len(df_all)}")
    print(f"  E distribution: {df_all['E'].value_counts().to_dict()}")
    for mod, cfg in MODALITIES.items():
        n = df_all[cfg["q"]].notna().sum()
        print(f"  {mod}: {n} rows with quality label  |  "
              f"Q: {df_all[cfg['q']].value_counts(dropna=True).to_dict()}")

    # ── Quality class helper ──────────────────────────────────────────────────
    # "GOOD" = all available channels are GOOD; "LOW" = at least one is BAD/NOISY
    def _qual_class(row):
        for cfg in MODALITIES.values():
            q = row[cfg["q"]]
            if pd.notna(q) and q != "GOOD":
                return "LOW"
        return "GOOD"

    df_all["_qual_class"] = df_all.apply(_qual_class, axis=1)
    strat_key = df_all["E"] + "_" + df_all["_qual_class"]

    print(f"\n  Quality distribution:  "
          f"all-GOOD={( df_all['_qual_class']=='GOOD').sum()}  "
          f"lower={(df_all['_qual_class']=='LOW').sum()}")

    # ── Stratified train / test split (E × quality class) ────────────────────
    train_df, test_df = train_test_split(
        df_all, test_size=0.2, random_state=42, stratify=strat_key
    )
    train_df = train_df.drop(columns=["_qual_class"]).reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)          # keep _qual_class for split below

    # ── Split test → test1 (all GOOD) / test2 (lower quality) ────────────────
    test1_df = test_df[test_df["_qual_class"] == "GOOD"].drop(columns=["_qual_class"]).reset_index(drop=True)
    test2_df = test_df[test_df["_qual_class"] == "LOW" ].drop(columns=["_qual_class"]).reset_index(drop=True)
    test_df  = test_df.drop(columns=["_qual_class"]).reset_index(drop=True)

    print(f"\n  train: {len(train_df)} rows")
    print(f"  test:  {len(test_df)} rows  "
          f"(test1 all-GOOD: {len(test1_df)},  test2 lower-quality: {len(test2_df)})\n")

    # ── Joint training over all modalities ────────────────────────────────────
    params = train_joint(train_df, use_quality)

    # Print high-level learned parameters
    e_vals = params["E_probs"].tolist()
    print(f"\n  E_probs: [{', '.join(f'{E_STATES[i]}={v:.3f}' for i, v in enumerate(e_vals))}]")
    for mod, cfg in MODALITIES.items():
        q_name = cfg["q"]
        q_vals = params[f"{q_name}_probs"].tolist()
        print(f"  {q_name}_probs: [{', '.join(f'{Q_STATES[i]}={v:.3f}' for i, v in enumerate(q_vals))}]")

    e_prior = params["E_probs"].detach().numpy()

    # ── KL information-gain matrix ────────────────────────────────────────────
    print("\nKL information-gain matrix  I = E[KL(P(E|X_m,Q) ‖ P(E|Q))]:")
    kl_mat = compute_kl_matrix(train_df, params, e_prior, use_quality)
    print(kl_mat.round(2).to_string())

    print("\nKL information-gain matrix per emotion  I = E[KL(P(E|X_m,Q) ‖ P(E|Q)) | E_true=e]:")
    kl_per_e = compute_kl_matrix_per_emotion(train_df, params, e_prior, use_quality)
    for e_name, mat in kl_per_e.items():
        print(f"\n  E = {e_name}:")
        print(mat.round(2).to_string())

    # ── Evaluation helper ─────────────────────────────────────────────────────
    def evaluate(df: pd.DataFrame, label: str) -> None:
        if len(df) == 0:
            print(f"\n=== {label} — brak próbek ===")
            return
        print(f"\n=== {label} ({len(df)} próbek) ===")
        res = predict_E(df, params, e_prior, use_quality)
        acc = accuracy_score(res["E_true"], res["E_pred"])
        print(f"\nAccuracy: {acc:.3f}  ({int(acc * len(res))}/{len(res)} correct)\n")
        print(classification_report(res["E_true"], res["E_pred"],
                                    target_names=E_STATES, zero_division=0))

    # ── Testing ───────────────────────────────────────────────────────────────
    evaluate(test_df,  "Test — cały zbiór testowy")
    evaluate(test1_df, "Test1 — tylko próbki all-GOOD")
    evaluate(test2_df, "Test2 — próbki niższej jakości")