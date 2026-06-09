import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

E_STATES = ["Angry", "Sad", "Happy", "Calm"]
Q_STATES = ["BAD", "NOISY", "GOOD"]

E = np.random.choice(E_STATES, size=n, p=[0.2, 0.2, 0.35, 0.25])
Q = np.random.choice(Q_STATES, size=n, p=[0.25, 0.35, 0.40])

Q_ENC = {"BAD": 0, "NOISY": 1, "GOOD": 2}
E_ENC = {"Angry": 0, "Sad": 1, "Happy": 2, "Calm": 3}

mean_V = 200 * np.array([Q_ENC[q] for q in Q]) + 80 * np.array([E_ENC[e] for e in E]) + 50
V = np.clip(np.random.normal(loc=mean_V, scale=60.0), 0, 1000)

df = pd.DataFrame({"E": E, "Q": Q, "V": V})
df.to_csv("dane.csv", index=False)
print(f"Saved dane.csv  ({len(df)} rows)")
print(df.head(8).to_string(index=False))
