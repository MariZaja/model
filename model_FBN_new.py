"""
Model FBN do klasyfikacji emocji — pgmpy FunctionalBayesianNetwork.

Dwa warianty grafu (wybór przez flagę --no-quality):

  Domyślnie (z quality):          --no-quality (bez quality):
    E ──► V  ◄── Q_audio            E ──► V
    E ──► V  ◄── Q_video            (Q węzłów brak w grafie)
    E ──► V  ◄── Q_eeg
    Q niezależne od E

Uczenie: model.fit(train_df, estimator="SVI") — pgmpy uruchamia Pyro SVI
Predykcja: ręczna inferencja bayesowska przez scipy_norm.logpdf
"""

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from scipy.stats import norm as scipy_norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from pgmpy.global_vars import config
config.set_backend("torch")

from pgmpy.models import FunctionalBayesianNetwork
from pgmpy.factors.hybrid import FunctionalCPD


# ── Encodings ──────────────────────────────────────────────────────────────────

E_STATES = ["Angry", "Sad", "Happy", "Calm"]
Q_STATES = ["BAD", "NOISY", "GOOD"]

E_BASE = "Calm"
Q_BASE = "GOOD"

E_ENC = {s: i for i, s in enumerate(E_STATES)}
Q_ENC = {s: i for i, s in enumerate(Q_STATES)}
E_DEC = {i: s for s, i in E_ENC.items()}
Q_DEC = {i: s for s, i in Q_ENC.items()}

EMOTION_MAP = {"Anger": "Angry", "Sadness": "Sad", "Happiness": "Happy", "Calm": "Calm"}

V_AUDIO = [f"V{k}" for k in range(1,  21)]   # V1–V20
V_VIDEO = [f"V{k}" for k in range(21, 41)]   # V21–V40
V_EEG   = [f"V{k}" for k in range(41, 61)]   # V41–V60

MODALITIES = {
    "audio": {"q": "Q_audio", "v_cols": V_AUDIO},
    "video": {"q": "Q_video", "v_cols": V_VIDEO},
    "eeg":   {"q": "Q_eeg",   "v_cols": V_EEG},
}

N_E = len(E_STATES)   # 4
N_Q = len(Q_STATES)   # 3


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(data_dir: str = "dane") -> pd.DataFrame:
    dq = pd.read_csv("prep/e01_data_quality_prep.csv")
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


# ── Przygotowanie DataFrame do fit() ──────────────────────────────────────────

def prepare_fit_df(df: pd.DataFrame, use_quality: bool) -> pd.DataFrame:
    """Koduje E i Q na int. Jeśli use_quality=False, kolumny Q są pomijane."""
    out = df.copy()
    out["E"] = out["E"].map(E_ENC)
    if use_quality:
        for cfg in MODALITIES.values():
            q_col = cfg["q"]
            out[q_col] = out[q_col].map(
                lambda x: Q_ENC[x] if (pd.notna(x) and x in Q_ENC) else np.nan
            )
    return out

def expand_reference_effect(
    raw: torch.Tensor,
    n_states: int,
    base_idx: int,
) -> torch.Tensor:
    """
    Zamienia wektor długości n_states - 1 na pełny wektor efektów,
    w którym efekt klasy bazowej wynosi 0.
    """
    parts = []
    raw_i = 0

    for i in range(n_states):
        if i == base_idx:
            parts.append(torch.tensor(0.0, dtype=raw.dtype, device=raw.device))
        else:
            parts.append(raw[raw_i])
            raw_i += 1

    return torch.stack(parts)

def expand_reference_effect_np(
    raw: np.ndarray,
    n_states: int,
    base_idx: int,
) -> np.ndarray:
    full = np.zeros(n_states)
    raw_i = 0

    for i in range(n_states):
        if i == base_idx:
            full[i] = 0.0
        else:
            full[i] = raw[raw_i]
            raw_i += 1

    return full

# ── CPD — węzeł E (wspólny dla obu modeli) ────────────────────────────────────

def cpd_fn_E(_parents):
    """Marginalny rozkład emocji P(E) — uczony jako simplex."""
    e_probs = pyro.param(
        "E_probs",
        torch.ones(N_E) / N_E,
        constraint=constraints.simplex,
    )
    return dist.Categorical(probs=e_probs)


# ── CPD — węzeł Q (tylko w modelu z quality) ──────────────────────────────────

def make_cpd_fn_Q(q_name: str):
    """Marginalny rozkład jakości P(Q) — Q niezależne od E."""
    def fn(_parents):
        q_probs = pyro.param(
            f"{q_name}_probs",
            torch.ones(N_Q) / N_Q,
            constraint=constraints.simplex,
        )
        return dist.Categorical(probs=q_probs)
    return fn


# ── CPD — węzeł V z quality: mu = beta0 + betaE[E] + betaQ[Q] ────────────────

def make_cpd_fn_V_with_quality(v_name: str, q_name: str):
    """V zależy od E i Q. Rodzice: [E, Q_mod]."""
    def fn(parents):
        beta_0 = pyro.param(f"{v_name}_beta_0", torch.tensor(0.0))

        beta_E_raw = pyro.param(
            f"{v_name}_beta_E_raw",
            torch.zeros(N_E - 1),
        )

        beta_Q_raw = pyro.param(
            f"{v_name}_beta_Q_raw",
            torch.zeros(N_Q - 1),
        )

        sigma = pyro.param(
            f"{v_name}_sigma",
            torch.tensor(1.0),
            constraint=constraints.positive,
        )

        e_base_idx = E_ENC[E_BASE]
        q_base_idx = Q_ENC[Q_BASE]

        beta_E = expand_reference_effect(beta_E_raw, N_E, e_base_idx)
        beta_Q = expand_reference_effect(beta_Q_raw, N_Q, q_base_idx)

        e_idx = parents["E"].long()
        q_idx = parents[q_name].long()

        mu = beta_0 + beta_E[e_idx] + beta_Q[q_idx]

        return dist.Normal(mu, sigma)
    return fn


# ── CPD — węzeł V bez quality: mu = beta0 + betaE[E] ─────────────────────────

def make_cpd_fn_V_no_quality(v_name: str):
    """V zależy tylko od E. Rodzice: [E]."""
    def fn(parents):
        beta_0 = pyro.param(f"{v_name}_beta_0", torch.tensor(0.0))

        beta_E_raw = pyro.param(
            f"{v_name}_beta_E_raw",
            torch.zeros(N_E - 1),
        )

        sigma = pyro.param(
            f"{v_name}_sigma",
            torch.tensor(1.0),
            constraint=constraints.positive,
        )

        e_base_idx = E_ENC[E_BASE]
        beta_E = expand_reference_effect(beta_E_raw, N_E, e_base_idx)

        e_idx = parents["E"].long()
        mu = beta_0 + beta_E[e_idx]

        return dist.Normal(mu, sigma)
    return fn


# ── Budowanie modelu ───────────────────────────────────────────────────────────

def build_model(use_quality: bool) -> FunctionalBayesianNetwork:
    """
    Buduje jeden z dwóch wariantów FBN:

    use_quality=True  → E──►V◄──Q  (krawędzie E→V i Q→V, Q bez rodziców)
    use_quality=False → E──►V       (tylko krawędzie E→V, brak węzłów Q)
    """
    edges = []
    for cfg in MODALITIES.values():
        q_col = cfg["q"]
        for v_name in cfg["v_cols"]:
            edges.append(("E", v_name))          # E ──► V  (oba modele)
            if use_quality:
                edges.append((q_col, v_name))    # Q ──► V  (tylko with-quality)

    model = FunctionalBayesianNetwork(edges)

    # Węzeł E — zawsze
    model.add_cpds(FunctionalCPD("E", fn=cpd_fn_E))

    for cfg in MODALITIES.values():
        q_col = cfg["q"]

        if use_quality:
            # Węzeł Q — tylko w modelu z quality
            model.add_cpds(
                FunctionalCPD(q_col, fn=make_cpd_fn_Q(q_col))
            )

        for v_name in cfg["v_cols"]:
            if use_quality:
                model.add_cpds(
                    FunctionalCPD(
                        v_name,
                        fn=make_cpd_fn_V_with_quality(v_name, q_col),
                        parents=["E", q_col],
                    )
                )
            else:
                model.add_cpds(
                    FunctionalCPD(
                        v_name,
                        fn=make_cpd_fn_V_no_quality(v_name),
                        parents=["E"],
                    )
                )

    assert model.check_model(), "Graf FBN jest niepoprawny — sprawdź krawędzie i CPD!"
    return model


# ── Uczenie ───────────────────────────────────────────────────────────────────

def train(
    model: FunctionalBayesianNetwork,
    train_df: pd.DataFrame,
    use_quality: bool,
    num_steps: int = 5000,
    lr: float = 0.005,
    seed: int = 7,
) -> dict[str, torch.Tensor]:
    """
    Dopasowuje model do danych treningowych przez SVI (Pyro pod spodem).
    Zwraca słownik parametrów z pyro.get_param_store().
    """
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    fit_df = prepare_fit_df(train_df, use_quality)

    # Kolumny wymagane przez dany wariant modelu
    q_cols    = [cfg["q"] for cfg in MODALITIES.values()]
    all_v_cols = V_AUDIO + V_VIDEO + V_EEG
    drop_cols  = (q_cols + all_v_cols) if use_quality else all_v_cols
    fit_df = fit_df.dropna(subset=drop_cols).reset_index(drop=True)

    mode_label = "z quality (E→V◄Q)" if use_quality else "bez quality (E→V)"
    print(f"Uczenie FBN [{mode_label}] na {len(fit_df)} oknach…")

    params = model.fit(
        fit_df,
        estimator="SVI",
        optimizer=pyro.optim.Adam({"lr": lr}),
        num_steps=num_steps,
        seed=seed,
    )

    return {k: v.detach().clone() for k, v in params.items()}


# ── Inferencja P(E | V, Q) ────────────────────────────────────────────────────

def predict_E(
    test_df: pd.DataFrame,
    params: dict[str, torch.Tensor],
    e_prior: np.ndarray,
    use_quality: bool,
) -> pd.DataFrame:
    records = []
    for _, row in test_df.iterrows():
        log_liks = np.zeros(len(E_STATES))

        for cfg in MODALITIES.values():

            if use_quality:
                q_col = cfg["q"]
                if pd.isna(row[q_col]):
                    continue
                q = int(Q_ENC[row[q_col]])
            else:
                q = None

            for v_name in cfg["v_cols"]:
                v_val = row[v_name]
                if pd.isna(v_val):
                    continue
                b0  = params[f"{v_name}_beta_0"].item()
                bE_raw = params[f"{v_name}_beta_E_raw"].detach().numpy()
                bE = expand_reference_effect_np(
                    bE_raw,
                    N_E,
                    E_ENC[E_BASE],
                )
                sig = params[f"{v_name}_sigma"].item()
                if use_quality:
                    bQ_raw = params[f"{v_name}_beta_Q_raw"].detach().numpy()
                    bQ = expand_reference_effect_np(
                        bQ_raw,
                        N_Q,
                        Q_ENC[Q_BASE],
                    )
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


# ── KL information-gain matrix ────────────────────────────────────────────────

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
                bE_raw = params[f"{v_name}_beta_E_raw"].detach().numpy()
                bE = expand_reference_effect_np(
                    bE_raw,
                    N_E,
                    E_ENC[E_BASE],
                )
                sig = params[f"{v_name}_sigma"].item()
                if use_quality:
                    bQ_raw = params[f"{v_name}_beta_Q_raw"].detach().numpy()
                    bQ = expand_reference_effect_np(
                        bQ_raw,
                        N_Q,
                        Q_ENC[Q_BASE],
                    )
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



# ── KL information-gain matrix per emocja ────────────────────────────────────

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
                bE_raw = params[f"{v_name}_beta_E_raw"].detach().numpy()
                bE = expand_reference_effect_np(
                    bE_raw,
                    N_E,
                    E_ENC[E_BASE],
                )
                sig = params[f"{v_name}_sigma"].item()
                if use_quality:
                    bQ_raw = params[f"{v_name}_beta_Q_raw"].detach().numpy()
                    bQ = expand_reference_effect_np(
                        bQ_raw,
                        N_Q,
                        Q_ENC[Q_BASE],
                    )
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
    parser = argparse.ArgumentParser(description="FBN klasyfikacja emocji")
    parser.add_argument(
        "--no-quality", action="store_true",
        help="Użyj modelu bez węzłów Q (tylko E→V). Domyślnie: z quality (E→V◄Q).",
    )
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    use_quality = not args.no_quality

    print(f"Tryb: {'z quality (E→V◄Q)' if use_quality else 'bez quality (E→V)'}\n")

    print("Wczytywanie danych…")
    df_all = load_data()
    print(f"  Łącznie okien: {len(df_all)}")
    print(f"  Rozkład E: {df_all['E'].value_counts().to_dict()}")
    for cfg in MODALITIES.values():
        n = df_all[cfg["q"]].notna().sum()
        print(f"  {cfg['q']}: {n} okien z etykietą jakości  "
              f"| {df_all[cfg['q']].value_counts(dropna=True).to_dict()}")

    # Stratyfikowany podział (E × klasa jakości)
    def _qual_class(row):
        for cfg in MODALITIES.values():
            q = row[cfg["q"]]
            if pd.notna(q) and q != "GOOD":
                return "LOW"
        return "GOOD"

    df_all["_qual_class"] = df_all.apply(_qual_class, axis=1)
    strat_key = df_all["E"] + "_" + df_all["_qual_class"]

    train_df, test_df = train_test_split(
        df_all, test_size=0.2, random_state=42, stratify=strat_key
    )
    train_df = train_df.drop(columns=["_qual_class"]).reset_index(drop=True)
    test1_df = test_df[test_df["_qual_class"] == "GOOD"].drop(columns=["_qual_class"]).reset_index(drop=True)
    test2_df = test_df[test_df["_qual_class"] == "LOW" ].drop(columns=["_qual_class"]).reset_index(drop=True)
    test_df  = test_df.drop(columns=["_qual_class"]).reset_index(drop=True)

    print(f"\n  Train: {len(train_df)} | Test: {len(test_df)} "
          f"(all-GOOD: {len(test1_df)}, lower-quality: {len(test2_df)})\n")

    # Buduj właściwy wariant modelu i ucz
    model  = build_model(use_quality)
    params = train(model, train_df, use_quality, num_steps=args.steps)

    e_vals = params["E_probs"].tolist()
    print(f"\n  E_probs: [{', '.join(f'{E_STATES[i]}={v:.3f}' for i, v in enumerate(e_vals))}]")
    e_prior = params["E_probs"].detach().numpy()

    # KL matrix
    print("\nMacierz KL  I(mod, Q):")
    kl_mat = compute_kl_matrix(train_df, params, e_prior, use_quality)
    print(kl_mat.round(3).to_string())

    # KL matrix per emocja
    print("Macierz KL per emocja  I = E[KL(P(E|X_m,Q) ‖ P(E)) | E_true=e]:")
    kl_per_e = compute_kl_matrix_per_emotion(train_df, params, e_prior, use_quality)
    for e_name, mat in kl_per_e.items():
        print(f"E = {e_name}:")
        print(mat.round(3).to_string())

    # Ewaluacja
    def evaluate(df: pd.DataFrame, label: str) -> None:
        if len(df) == 0:
            print(f"\n=== {label} — brak próbek ===")
            return
        print(f"\n=== {label} ({len(df)} próbek) ===")
        res = predict_E(df, params, e_prior, use_quality)
        acc = accuracy_score(res["E_true"], res["E_pred"])
        print(f"Accuracy: {acc:.3f}  ({int(acc * len(res))}/{len(res)})\n")
        print(classification_report(
            res["E_true"],
            res["E_pred"],
            labels=E_STATES,
            target_names=E_STATES,
            zero_division=0,)
        )

    evaluate(test_df,  "Test — cały zbiór")
    evaluate(test1_df, "Test1 — all-GOOD")
    evaluate(test2_df, "Test2 — lower-quality")