"""
compare_issuer_level -- replicate the ORIGINAL setup (linear model trained on
one issuer's bonds, like Run Code Part 1's issuer filter) and race it against
the universe-trained LightGBM on that same issuer's bonds.

Three numbers per issuer pool:
  1. linear IN-SAMPLE   -- fit on the pool, score on the same bonds.
                           (This is likely where a ~5.6 bps figure comes from;
                           it flatters the model because it grades its own fit.)
  2. linear CROSS-VAL   -- 5-fold within the pool: predict bonds the fit never
                           saw. The honest version of the same approach.
  3. lgbm UNIVERSE      -- LightGBM trained on the whole universe EXCLUDING the
                           pool, predicting the pool. Honest, and needs no
                           issuer filter at all.

    python compare_issuer_level.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm  # noqa: E402

print("loading evals...")
df = mm.load_evals(HERE / 'ICE_Evals.csv')
df = mm.clean_universe(df)
df = mm.build_features(df)

numeric = [c for c in mm.NUMERIC_FEATURES if c in df.columns]
categorical = [c for c in mm.CATEGORICAL_FEATURES if c in df.columns]

TEXT_COLS = [c for c in ['primary_name_abbreviated', 'organization_master_id',
                         'conduit_obligor_name_id'] if c in df.columns]


def issuer_mask(pattern):
    m = pd.Series(False, index=df.index)
    for c in TEXT_COLS:
        m |= df[c].astype(str).str.contains(pattern, case=False, na=False, regex=False)
    return m


def make_linear():
    return Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('sc', StandardScaler())]), numeric),
            ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                  min_frequency=2, sparse_output=True), categorical),
        ])),
        ('model', Ridge(alpha=1.0)),
    ])


def bps(pred, actual):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(actual)) * 100))


X_lin_all = df[numeric + categorical].copy()
for c in categorical:
    X_lin_all[c] = X_lin_all[c].astype(str)
y_all = df['target_yield'].astype(float)

POOLS = {
    'LA MTA sales tax (Part 1 filter)': 'LOS ANG CY CA MET TRA AUT',
    'all LOS ANG issuers': 'LOS ANG',
    'all California transportation': None,  # built below
}

rows = []
for name, pat in POOLS.items():
    if pat is not None:
        m = issuer_mask(pat)
    else:
        m = (df.get('incorporated_state_code_desc').eq('California')
             & df.get('purpose_class_desc').astype(str).str.contains('Transport', case=False, na=False))
    n = int(m.sum())
    if n < 30:
        print(f"{name}: only {n} bonds, skipping")
        continue

    X_pool, y_pool = X_lin_all[m], y_all[m]

    # 1. linear, in-sample (grades its own fit)
    lin = make_linear().fit(X_pool, y_pool)
    insample = bps(lin.predict(X_pool), y_pool)

    # 2. linear, 5-fold cross-validated within the pool
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_err = []
    for tr, te in kf.split(X_pool):
        f = make_linear().fit(X_pool.iloc[tr], y_pool.iloc[tr])
        cv_err.append(np.abs(f.predict(X_pool.iloc[te]) - y_pool.iloc[te].to_numpy()) * 100)
    cv = float(np.mean(np.concatenate(cv_err)))

    # 3. lightgbm trained on the rest of the universe, predicting the pool
    X_gbm = mm._prep_matrix(df, numeric, categorical)
    gbm = LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127,
                        min_child_samples=30, subsample=0.9, subsample_freq=1,
                        colsample_bytree=0.9, random_state=0, verbose=-1)
    gbm.fit(X_gbm[~m], y_all[~m])
    gbm_err = bps(gbm.predict(X_gbm[m]), y_pool)

    rows.append({'pool': name, 'bonds': n,
                 'linear in-sample': round(insample, 1),
                 'linear cross-val': round(cv, 1),
                 'lgbm universe': round(gbm_err, 1)})
    print(f"{name} ({n:,} bonds): in-sample {insample:.1f} | "
          f"cross-val {cv:.1f} | lgbm {gbm_err:.1f} bps")

res = pd.DataFrame(rows).set_index('pool')
res.to_csv(HERE / 'issuer_level_comparison.csv')
print("\n=============== issuer-level verdict (MAE bps) ===============")
print(res.to_string())
