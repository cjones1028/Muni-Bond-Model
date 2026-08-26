"""Final candidate: drop <1yr from training (the cleaning that worked) +
par_dv01 feature (the feature that helped every wire). Graded on both
comparison slices + all wires."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from Wire_Parser import parse_wire

df = mm.load_evals(HERE / 'ICE_Evals.csv')
base = mm.clean_universe(df)
tenor = pd.to_numeric(base['days_to_maturity_30_360'], errors='coerce')
train = base[tenor >= 360]
print(f"training on {len(train):,} bonds (>=1yr), par_dv01 feature ON")

b = mm.train_yield_model(train, n_ensemble=1, seed=42)

dfF = mm.build_features(base)
rng = np.random.RandomState(42)
r = rng.rand(len(base))
d = pd.to_numeric(base['ice_dv01'], errors='coerce')
slices = {'>=1yr slice': (r < 0.2) & (tenor >= 360).to_numpy(),
          'dv01>=1 slice': (r < 0.2) & (d >= 1).fillna(False).to_numpy()}
for name, mask in slices.items():
    h = dfF[mask]
    X = mm._prep_matrix(h, b['numeric'], b['categorical'], b['categories'])
    err = np.abs(mm.predict_bundle(b, X) - h['target_yield'].to_numpy()) * 100
    print(f"holdout {name} ({mask.sum():,}): MAE {err.mean():.2f} | median {np.median(err):.2f}")

for wname, wfile, iss in [('LA', 'BAML Write-Up.txt', 'LOS ANG'),
                          ('NYC', 'NYC_TFA_wire.txt', 'CITY TRANSITIONAL FIN'),
                          ('PSU', 'psu_wire.txt', 'PENNSYLVANIA STATE UNIVERSITY'),
                          ('PDX', 'pasted_wire.txt', 'PORTLAND ORE SWR')]:
    deal = parse_wire(open(HERE / wfile, encoding='utf-8').read())
    templ = mm.template_from(base, issuer_contains=iss)
    res = mm.price_wire(deal, b, templ, concession_bps=10.5)
    print(f">>> {wname}: mean {res['Error (bps)'].abs().mean():.1f} | "
          f"front {res['Error (bps)'].iloc[0]:+.1f}")
