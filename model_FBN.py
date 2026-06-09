"""
Functional Bayesian Network (FBN) — pgmpy + Pyro
Structure:  Q -> V,  E -> V

Variables:
  E: emotion  — categorical {Angry=0, Sad=1, Happy=2, Calm=3}
  Q: quality  — categorical {BAD=0, NOISY=1, GOOD=2}
  V: feature  — continuous [0, 1000]

Functional CPDs:
  P(E) = Categorical(probs_E)
  P(Q) = Categorical(probs_Q)
  P(V | E, Q) = Normal( beta_0 + beta_E * E + beta_Q * Q , sigma )
"""

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro import param
from torch import tensor
from torch.distributions import constraints
from scipy.stats import norm as scipy_norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import pgmpy
pgmpy.config.set_backend("torch")

from pgmpy.models import FunctionalBayesianNetwork
from pgmpy.factors.hybrid import FunctionalCPD

# ── Encodings ─────────────────────────────────────────────────────────────────
E_STATES = ["Angry", "Sad", "Happy", "Calm"]
Q_STATES = ["BAD", "NOISY", "GOOD"]

E_ENC = {s: i for i, s in enumerate(E_STATES)}
Q_ENC = {s: i for i, s in enumerate(Q_STATES)}
E_DEC = {i: s for s, i in E_ENC.items()}
Q_DEC = {i: s for s, i in Q_ENC.items()}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        E=df["E"].map(E_ENC).astype(float),
        Q=df["Q"].map(Q_ENC).astype(float),
    )


# ── Functional CPD definitions ────────────────────────────────────────────────

def cpd_E_fn(parents):
    probs = param(
        "E_probs",
        tensor([0.25, 0.25, 0.25, 0.25]),
        constraint=constraints.simplex,
    )
    return dist.Categorical(probs=probs)


def cpd_Q_fn(parents):
    probs = param(
        "Q_probs",
        tensor([1/3, 1/3, 1/3]),
        constraint=constraints.simplex,
    )
    return dist.Categorical(probs=probs)


def cpd_V_fn(parents):
    beta_0 = param("V_beta_0", tensor(0.0))
    # one offset per category — no ordering assumed
    beta_E = param("V_beta_E", tensor([0.0,   80.0, 160.0, 240.0]))  # len = |E_STATES|
    beta_Q = param("V_beta_Q", tensor([0.0,  200.0, 400.0]))         # len = |Q_STATES|
    sigma  = param("V_sigma",  tensor(60.0), constraint=constraints.positive)

    e_idx = parents["E"].long()
    q_idx = parents["Q"].long()
    mu = beta_0 + beta_E[e_idx] + beta_Q[q_idx]
    return dist.Normal(mu, sigma)


# ── Posterior P(E | V, Q) using Bayes' theorem ───────────────────────────────

def predict_E(df_enc: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    For each row with observed V and Q, compute P(E | V, Q) and return MAP label.

    P(E=e | V, Q) ∝ P(V | E=e, Q) * P(E=e)
    P(V | E=e, Q)  = Normal(beta_0 + beta_E[e] + beta_Q[q], sigma)
    """
    beta_0  = params["V_beta_0"].item()
    beta_E  = params["V_beta_E"].detach().numpy()   # vector, one value per E category
    beta_Q  = params["V_beta_Q"].detach().numpy()   # vector, one value per Q category
    sigma   = params["V_sigma"].item()
    e_prior = params["E_probs"].detach().numpy()

    records = []
    for _, row in df_enc.iterrows():
        v, q = row["V"], int(row["Q"])

        posteriors = np.array([
            scipy_norm.pdf(v, beta_0 + beta_E[e] + beta_Q[q], sigma) * e_prior[e]
            for e in range(len(E_STATES))
        ])
        posteriors /= posteriors.sum() + 1e-300

        pred_idx = int(np.argmax(posteriors))
        records.append({
            "E_true":  E_DEC[int(row["E"])],
            "E_pred":  E_DEC[pred_idx],
            "V":       v,
            "Q_label": Q_DEC[int(q)],
            **{f"P({E_STATES[e]})": f"{posteriors[e]:.3f}" for e in range(len(E_STATES))},
        })

    return pd.DataFrame(records)


# ── Build model ───────────────────────────────────────────────────────────────

def build_model() -> FunctionalBayesianNetwork:
    model = FunctionalBayesianNetwork([("Q", "V"), ("E", "V")])
    model.add_cpds(
        FunctionalCPD("E", fn=cpd_E_fn),
        FunctionalCPD("Q", fn=cpd_Q_fn),
        FunctionalCPD("V", fn=cpd_V_fn, parents=["E", "Q"]),
    )
    assert model.check_model()
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load ──────────────────────────────────────────────────────────────────
    df_raw = pd.read_csv("dane.csv")
    print(f"Loaded {len(df_raw)} rows.\n")

    # ── Train / test split ────────────────────────────────────────────────────
    df_train_raw, df_test_raw = train_test_split(df_raw, test_size=0.2, random_state=42)
    df_train = encode(df_train_raw)
    df_test  = encode(df_test_raw)
    print(f"Train: {len(df_train)} rows  |  Test: {len(df_test)} rows\n")

    # ── Build & train ─────────────────────────────────────────────────────────
    model = build_model()
    print("Model structure:", list(model.edges()))
    print("\nFitting on training set (SVI, 5000 steps)…")
    pyro.clear_param_store()
    params = model.fit(df_train, estimator="SVI", num_steps=5000, seed=42)

    print("\n=== Learned parameters ===")
    for name, val in params.items():
        if isinstance(val, torch.Tensor) and val.numel() > 1:
            labels = E_STATES if "E_" in name else Q_STATES if "Q_" in name else None
            if labels:
                formatted = ", ".join(f"{labels[i]}={v:.1f}" for i, v in enumerate(val.tolist()))
            else:
                formatted = ", ".join(f"{v:.3f}" for v in val.tolist())
            print(f"  {name}: [{formatted}]")
        else:
            v = val.item() if isinstance(val, torch.Tensor) else val
            print(f"  {name}: {v:.3f}")

    # ── Predict E on test set ─────────────────────────────────────────────────
    print("\n=== Predicting E on test set ===")
    results = predict_E(df_test, params)

    print(results[["E_true", "E_pred", "V", "Q_label",
                    "P(Angry)", "P(Sad)", "P(Happy)", "P(Calm)"]].to_string(index=False))

    # ── Evaluation ────────────────────────────────────────────────────────────
    acc = accuracy_score(results["E_true"], results["E_pred"])
    print(f"\nAccuracy: {acc:.3f}  ({int(acc*len(results))}/{len(results)} correct)\n")
    print(classification_report(results["E_true"], results["E_pred"], target_names=E_STATES))
