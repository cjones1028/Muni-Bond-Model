"""Boss suggestions 8/26 round 2: (a) analytic DV01 as an input feature,
(b) drop training bonds with quote-DV01 < 1 (the original Part 1 filter).
Four configs, one fair holdout (full-universe seed-42 slice, dv01>=1 & >=1yr
so every config is graded on the same bonds), plus all four live wires."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from Wire_Parser import parse_wire

df_all = mm.load_evals(HERE / 'ICE_Evals.csv')
base = mm.clean_universe(df_all)
filt = mm.clean_universe(df_all, min_dv01=1)

NO_DV01 = [f for f in mm.NUMERIC_FEATURES if f != 'par_dv01']

configs = [
    ('baseline (no dv01 feat, no filter)', base, NO_DV01),
    ('+par_dv01 feature',                  base, None),
    ('min_dv01=1 filter',                  filt, NO_DV01),
    ('BOTH (feature + filter)',            filt, None),
]

# fair grading slice: same bonds for every config
dfF = mm.build_features(base)
rng = np.random.RandomState(42)
d = pd.to_numeric(base['ice_dv01'], errors='coerce')
grade = (rng.rand(len(base)) < 0.2) & (d >= 1).fillna(False).to_numpy()
hold = dfF[grade]
print(f"grading slice: {grade.sum():,} bonds (dv01>=1)")

wires = [('LA', 'BAML Write-Up.txt', 'LOS ANG'),
         ('NYC', 'NYC_TFA_wire.txt', 'CITY TRANSITIONAL FIN'),
         ('PSU', 'psu_wire.txt', 'PENNSYLVANIA STATE UNIVERSITY'),
         ('PDX', 'pasted_wire.txt', 'PORTLAND ORE SWR')]

for label, dtrain, feats in configs:
    b = mm.train_yield_model(dtrain, numeric=feats, n_ensemble=1, seed=42)
    X = mm._prep_matrix(hold, b['numeric'], b['categorical'], b['categories'])
    err = np.abs(mm.predict_bundle(b, X) - hold['target_yield'].to_numpy()) * 100
    parts = []
    for wname, wfile, iss in wires:
        deal = parse_wire(open(HERE / wfile, encoding='utf-8').read())
        templ = mm.template_from(base, issuer_contains=iss)
        r = mm.price_wire(deal, b, templ, concession_bps=10.5)
        parts.append(f"{wname} {r['Error (bps)'].abs().mean():.1f}/{r['Error (bps)'].iloc[0]:+.0f}")
    print(f">>> {label}: holdout {err.mean():.2f}/{np.median(err):.2f} | "
          f"wires(mean/front): {' | '.join(parts)}")
