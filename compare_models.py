"""
compare_models -- head-to-head simulation: original-style LINEAR model vs the
new GRADIENT BOOSTING model, on identical data, features, and splits.

    python compare_models.py [n_rounds]

Each round: fresh random 80/20 train/test split of the cleaned ICE universe;
both models train on the same 80 and predict the same 20; error measured in
bps against ICE's mid yield. Repeated n_rounds times (default 12) so the
verdict comes from a distribution, not one lucky split.

The linear contender reconstructs the Curve_Analysis approach (statsmodels-
style regression on the Part 1 feature lists): median-imputed numerics +
one-hot categoricals + ridge regularization (plain OLS is degenerate with
thousands of one-hot columns; ridge is the charitable version).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm  # noqa: E402

N_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12

print("loading evals...")
df = mm.load_evals(HERE / 'ICE_Evals.csv')
df = mm.clean_universe(df)
df = mm.build_features(df)

numeric = [c for c in mm.NUMERIC_FEATURES if c in df.columns]
categorical = [c for c in mm.CATEGORICAL_FEATURES if c in df.columns]
y = df['target_yield'].astype(float).to_numpy()


def make_linear():
    return Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('sc', StandardScaler())]), numeric),
            ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                  min_frequency=25, sparse_output=True), categorical),
        ])),
        ('model', Ridge(alpha=1.0)),
    ])


def make_lgbm(seed):
    return LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127,
                         min_child_samples=30, subsample=0.9, subsample_freq=1,
                         colsample_bytree=0.9, random_state=seed, verbose=-1)


X_lin = df[numeric + categorical].copy()
for c in categorical:
    X_lin[c] = X_lin[c].astype(str)
X_gbm = mm._prep_matrix(df, numeric, categorical)

rows = []
for r in range(N_ROUNDS):
    rng = np.random.RandomState(1000 + r)
    test = rng.rand(len(df)) < 0.2
    t0 = time.time()

    lin = make_linear().fit(X_lin[~test], y[~test])
    lin_err = np.abs(lin.predict(X_lin[test]) - y[test]) * 100

    gbm = make_lgbm(1000 + r).fit(X_gbm[~test], y[~test])
    gbm_err = np.abs(gbm.predict(X_gbm[test]) - y[test]) * 100

    beat = float((gbm_err < lin_err).mean())
    rows.append({'round': r + 1,
                 'linear MAE': lin_err.mean(), 'linear median': np.median(lin_err),
                 'lgbm MAE': gbm_err.mean(), 'lgbm median': np.median(gbm_err),
                 'lgbm wins bond-level %': 100 * beat,
                 'secs': time.time() - t0})
    print(f"round {r+1:2d}/{N_ROUNDS}: linear {lin_err.mean():6.1f} bps | "
          f"lgbm {gbm_err.mean():5.1f} bps | lgbm better on "
          f"{100*beat:.0f}% of bonds | {rows[-1]['secs']:.0f}s")

res = pd.DataFrame(rows).set_index('round')
res.to_csv(HERE / 'model_comparison.csv')

print("\n================ VERDICT over", N_ROUNDS, "simulations ================")
print(res[['linear MAE', 'lgbm MAE', 'lgbm wins bond-level %']].round(1).to_string())
print()
lin_m, gbm_m = res['linear MAE'].mean(), res['lgbm MAE'].mean()
wins = int((res['lgbm MAE'] < res['linear MAE']).sum())
print(f"average MAE : linear {lin_m:.1f} bps  vs  lightgbm {gbm_m:.1f} bps")
print(f"round wins  : lightgbm {wins}/{N_ROUNDS}")
print(f"per-bond    : lightgbm closer on {res['lgbm wins bond-level %'].mean():.0f}% of bonds")
print(f"improvement : {lin_m - gbm_m:.1f} bps ({100*(lin_m-gbm_m)/lin_m:.0f}% error reduction)")
