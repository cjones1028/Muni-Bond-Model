"""
exp_prune -- can we drop half the columns and retrain faster without losing
accuracy? (Boss suggestion 8/26.)

1. Rank every feature by its recorded contribution (LightGBM gain) in the
   current production ensemble.
2. Retrain a single model on ALL features vs the TOP half, timing both.
3. Compare holdout accuracy, and re-price all four live wires as a veto.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from Wire_Parser import parse_wire

df = mm.load_evals(HERE / 'ICE_Evals.csv')
df = mm.clean_universe(df)

bundle = mm.load_bundle(HERE / 'model.joblib')
feats = bundle['numeric'] + bundle['categorical']
gain = np.mean([m.booster_.feature_importance(importance_type='gain')
                for m in bundle['models']], axis=0)
imp = pd.Series(gain, index=feats).sort_values(ascending=False)
share = imp / imp.sum() * 100

print("=== feature importance (share of total model gain, %) ===")
for name, s in share.items():
    print(f"  {s:6.2f}%  {name}")

keep_n = int(np.ceil(len(imp) / 2))
keep = list(imp.index[:keep_n])
drop = list(imp.index[keep_n:])
print(f"\nkeeping top {keep_n} features ({share.iloc[:keep_n].sum():.2f}% of gain), "
      f"dropping {len(drop)} ({share.iloc[keep_n:].sum():.2f}%)")

configs = {
    'ALL features': (bundle['numeric'], bundle['categorical']),
    f'TOP {keep_n} features': ([f for f in bundle['numeric'] if f in keep],
                               [f for f in bundle['categorical'] if f in keep]),
}

wires = [('LA', 'BAML Write-Up.txt', 'LOS ANG'),
         ('NYC', 'NYC_TFA_wire.txt', 'CITY TRANSITIONAL FIN'),
         ('PSU', 'psu_wire.txt', 'PENNSYLVANIA STATE UNIVERSITY'),
         ('PDX', 'pasted_wire.txt', 'PORTLAND ORE SWR')]

for label, (num, cat) in configs.items():
    t0 = time.time()
    b = mm.train_yield_model(df, numeric=num, categorical=cat, n_ensemble=1, seed=42)
    mins = (time.time() - t0) / 60
    wire_res = []
    for wname, wfile, iss in wires:
        deal = parse_wire(open(HERE / wfile, encoding='utf-8').read())
        templ = mm.template_from(df, issuer_contains=iss)
        r = mm.price_wire(deal, b, templ, concession_bps=10.5)
        wire_res.append(f"{wname} {r['Error (bps)'].abs().mean():.1f}")
    print(f">>> {label}: train {mins:.1f} min | holdout {b['mae_bps']:.2f}/"
          f"{b['median_bps']:.2f} | wires: {' | '.join(wire_res)}")
