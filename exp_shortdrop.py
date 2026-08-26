"""Does dropping sub-1-year bonds from TRAINING help? (Boss suggestion 8/26.)
Train single-seed models on all bonds vs bonds >=1yr; grade both on the SAME
holdout slice (>=1yr, so the comparison is fair) AND on all four live wires,
watching the shortest tranches specifically."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from Wire_Parser import parse_wire

df = mm.load_evals(HERE / 'ICE_Evals.csv')
df = mm.clean_universe(df)
tenor_days = pd.to_numeric(df['days_to_maturity_30_360'], errors='coerce')

wires = [('LA', 'BAML Write-Up.txt', 'LOS ANG'),
         ('NYC', 'NYC_TFA_wire.txt', 'CITY TRANSITIONAL FIN'),
         ('PSU', 'psu_wire.txt', 'PENNSYLVANIA STATE UNIVERSITY'),
         ('PDX', 'pasted_wire.txt', 'PORTLAND ORE SWR')]

configs = {'ALL bonds': df, 'DROP <1yr': df[tenor_days >= 360]}

for label, dtrain in configs.items():
    print(f"--- training on {len(dtrain):,} bonds ({label}) ---")
    b = mm.train_yield_model(dtrain, n_ensemble=1, seed=42)

    # fair grading: same >=1yr slice of the full-universe seed-42 holdout
    dfF = mm.build_features(df)
    rng = np.random.RandomState(42)
    test = (rng.rand(len(dfF)) < 0.2) & (tenor_days >= 360).to_numpy()
    h = dfF[test]
    X = mm._prep_matrix(h, b['numeric'], b['categorical'], b['categories'])
    err = np.abs(mm.predict_bundle(b, X) - h['target_yield'].to_numpy()) * 100
    print(f"holdout >=1yr ({test.sum():,} bonds): MAE {err.mean():.2f} | median {np.median(err):.2f}")

    for wname, wfile, iss in wires:
        deal = parse_wire(open(HERE / wfile, encoding='utf-8').read())
        templ = mm.template_from(df, issuer_contains=iss)
        r = mm.price_wire(deal, b, templ, concession_bps=10.5)
        front = r['Error (bps)'].iloc[0]
        print(f"  {wname}: mean {r['Error (bps)'].abs().mean():.1f} | shortest tranche {front:+.1f}")
