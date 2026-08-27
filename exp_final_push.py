"""Last legitimate accuracy levers, measured on the PERSISTED leak-free
holdout (same bonds for every config):

A. Ensemble 3 -> 5: train seeds 45,46 and append to the production trio.
B. Tree cap 8000 -> 12000: models still improving at the cap; one single-seed
   pair (seed 42 @8000 exists in the bundle) vs seed 42 @12000.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm

bundle = mm.load_bundle(HERE / 'model.joblib')
hc = set(bundle['holdout_cusips'])
df = mm.stacked_frame(HERE)
dfF = mm.build_features(df)
hold = dfF[dfF.index.isin(hc)]
X = mm._prep_matrix(hold, bundle['numeric'], bundle['categorical'], bundle['categories'])
y = hold['target_yield'].to_numpy()


def mae(models):
    p = np.mean([m.predict(X) for m in models], axis=0)
    e = np.abs(p - y) * 100
    return e.mean(), np.median(e)


m3 = mae(bundle['models'])
print(f"baseline 3-ensemble @8000: MAE {m3[0]:.2f} | median {m3[1]:.2f}")

extra = []
for seed in (45, 46):
    b = mm.train_yield_model(df, n_ensemble=1, seed=seed)
    extra.append(b['models'][0])
m5 = mae(bundle['models'] + extra)
print(f">>> A. 5-ensemble @8000:  MAE {m5[0]:.2f} | median {m5[1]:.2f} "
      f"(delta {m5[0]-m3[0]:+.2f}/{m5[1]-m3[1]:+.2f})")

m1_8k = mae([bundle['models'][0]])
b12 = mm.train_yield_model(df, n_ensemble=1, seed=42, n_estimators=12000)
m1_12k = mae(b12['models'])
print(f">>> B. single seed42: @8000 MAE {m1_8k[0]:.2f}/{m1_8k[1]:.2f} vs "
      f"@12000 {m1_12k[0]:.2f}/{m1_12k[1]:.2f} "
      f"(delta {m1_12k[0]-m1_8k[0]:+.2f}/{m1_12k[1]-m1_8k[1]:+.2f}) "
      f"| best_iter {b12['models'][0].best_iteration_}")
