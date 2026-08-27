"""Correct 5-ensemble test: ONE training call (one CUSIP split, seed 42),
five members differing only in tree randomness. Compare members[:3] vs all
five on the same untouched holdout -- the leak-free version of the earlier
(contaminated) +2-seed test."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm

df = mm.stacked_frame(HERE)
b = mm.train_yield_model(df, n_ensemble=5, seed=42)

dfF = mm.build_features(df)
hold = dfF[dfF.index.isin(set(b['holdout_cusips']))]
X = mm._prep_matrix(hold, b['numeric'], b['categorical'], b['categories'])
y = hold['target_yield'].to_numpy()


def mae(models):
    e = np.abs(np.mean([m.predict(X) for m in models], axis=0) - y) * 100
    return e.mean(), np.median(e)


m3, m5 = mae(b['models'][:3]), mae(b['models'])
print(f">>> 3 members: MAE {m3[0]:.2f} | median {m3[1]:.2f}")
print(f">>> 5 members: MAE {m5[0]:.2f} | median {m5[1]:.2f} "
      f"(delta {m5[0]-m3[0]:+.2f}/{m5[1]-m3[1]:+.2f})")
mm.save_bundle(b, HERE / 'model_ens5_candidate.joblib')
print("candidate saved as model_ens5_candidate.joblib (NOT production)")
