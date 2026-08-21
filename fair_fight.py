"""
fair_fight -- the properly matched race on the old approach's home turf.

For each issuer pool, 5-fold cross-validation where BOTH models get the same
information per fold:
  * linear: trains on 4/5 of the pool (the original recipe)
  * lgbm:   trains on the whole universe PLUS the same 4/5 of the pool
            (production setup: issuer's other bonds in training, issuer id
            as a feature)
Both predict the same held-out 1/5. Repeat over all folds.

    python fair_fight.py
"""

import sys
import warnings
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

warnings.filterwarnings('ignore')

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


X_lin = df[numeric + categorical].copy()
for c in categorical:
    X_lin[c] = X_lin[c].astype(str)
X_gbm = mm._prep_matrix(df, numeric, categorical)
y = df['target_yield'].astype(float)

POOLS = {
    'LA MTA sales tax (Part 1 filter)': issuer_mask('LOS ANG CY CA MET TRA AUT'),
    'all LOS ANG issuers': issuer_mask('LOS ANG'),
    'all California transportation':
        (df['incorporated_state_code_desc'].eq('California')
         & df['purpose_class_desc'].astype(str).str.contains('Transport', case=False, na=False)),
}

rows = []
for name, m in POOLS.items():
    pool_idx = np.where(m.to_numpy())[0]
    n = len(pool_idx)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    lin_err, gbm_err = [], []
    for tr, te in kf.split(pool_idx):
        tr_i, te_i = pool_idx[tr], pool_idx[te]

        lin = make_linear().fit(X_lin.iloc[tr_i], y.iloc[tr_i])
        lin_err.append(np.abs(lin.predict(X_lin.iloc[te_i]) - y.iloc[te_i].to_numpy()) * 100)

        train_mask = np.ones(len(df), bool)
        train_mask[te_i] = False
        gbm = LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127,
                            min_child_samples=30, subsample=0.9, subsample_freq=1,
                            colsample_bytree=0.9, random_state=0, verbose=-1)
        gbm.fit(X_gbm[train_mask], y[train_mask])
        gbm_err.append(np.abs(gbm.predict(X_gbm.iloc[te_i]) - y.iloc[te_i].to_numpy()) * 100)

    lin_mae = float(np.mean(np.concatenate(lin_err)))
    gbm_mae = float(np.mean(np.concatenate(gbm_err)))
    rows.append({'pool': name, 'bonds': n,
                 'linear (old recipe)': round(lin_mae, 1),
                 'lgbm (production)': round(gbm_mae, 1)})
    print(f"{name} ({n:,} bonds): linear {lin_mae:.1f} bps | lgbm {gbm_mae:.1f} bps")

res = pd.DataFrame(rows).set_index('pool')
res.to_csv(HERE / 'fair_fight_results.csv')
print("\n=============== fair fight (5-fold CV MAE, bps) ===============")
print(res.to_string())
