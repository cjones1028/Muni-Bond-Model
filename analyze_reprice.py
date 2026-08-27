"""Grade the model's Penn State call against the actual repricing:
did the tranches the model flagged rich get their yields raised, and the
fair/cheap ones get cut? Also: how close are the PROPOSED yields to the
model's fair values?"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from Wire_Parser import parse_wire

# repricing wire (Barclays, 8/26): subscription and proposed adjustments
REPRICE = {  # maturity: (subscription_x, adj_bps, proposed_yield)
 '09/01/2027': (0.4, +5, 2.53), '09/01/2028': (0.6, +5, 2.62),
 '09/01/2029': (0.5, +5, 2.72), '09/01/2030': (0.0, +5, 2.85),
 '09/01/2031': (1.0, +2, 2.93), '09/01/2032': (0.5, +2, 3.04),
 '09/01/2033': (0.5, +2, 3.15), '09/01/2034': (2.0, 0, 3.28),
 '09/01/2035': (3.0, -2, 3.37), '09/01/2036': (3.3, -2, 3.48),
 '09/01/2037': (2.1, -2, 3.61), '09/01/2038': (2.2, -2, 3.73),
 '09/01/2039': (2.4, -2, 3.87), '09/01/2040': (2.4, -2, 3.97),
 '09/01/2041': (3.2, -2, 4.03), '09/01/2042': (1.0, 0, 4.10),
 '09/01/2043': (0.1, 0, 4.15), '09/01/2044': (0.6, 0, 4.21),
 '09/01/2045': (0.8, 0, 4.28), '09/01/2046': (1.5, 0, 4.38),
 '09/01/2051': (2.0, -1, 4.60), '09/01/2056': (2.5, -2, 4.81),
}

deal = parse_wire(open(HERE / 'psu_wire.txt', encoding='utf-8').read())
df = pd.read_parquet(HERE / 'template_cache.parquet')
bundle = mm.load_bundle(HERE / 'model.joblib')
templ = mm.template_from(df, issuer_contains='PENNSYLVANIA STATE UNIVERSITY')
from calibration import concession
res = mm.price_wire(deal, bundle, templ, concession_bps=concession())

rows = []
for mat, r in res.iterrows():
    if mat not in REPRICE:
        continue
    sub, adj, py = REPRICE[mat]
    rows.append({'mat': mat, 'model_err': r['Error (bps)'], 'sub': sub,
                 'adj': adj, 'new_err': round((r['Model Yield'] - py) * 100, 1)})
t = pd.DataFrame(rows)

r_adj = np.corrcoef(t['model_err'], t['adj'])[0, 1]
r_sub = np.corrcoef(t['model_err'], t['sub'])[0, 1]
print(f"correlation model-error vs yield ADJUSTMENT: {r_adj:+.2f}")
print(f"correlation model-error vs SUBSCRIPTION:    {r_sub:+.2f}")

rich = t[t['model_err'] > 5]
fair = t[t['model_err'] <= 5]
print(f"\nmodel-flagged RICH (>5bp): avg subscription {rich['sub'].mean():.1f}x, "
      f"avg adjustment {rich['adj'].mean():+.1f} bps  (n={len(rich)})")
print(f"model-called FAIR/CHEAP:   avg subscription {fair['sub'].mean():.1f}x, "
      f"avg adjustment {fair['adj'].mean():+.1f} bps  (n={len(fair)})")
print(f"\nerror vs PRELIMINARY yields: mean abs {t['model_err'].abs().mean():.1f} bps")
print(f"error vs PROPOSED    yields: mean abs {t['new_err'].abs().mean():.1f} bps")
print("\nfull scorecard (model gap vs preliminary -> vs proposed):")
t['moved_toward_model'] = np.where(t['new_err'].abs() < t['model_err'].abs() - 0.05, 'YES',
                          np.where(t['new_err'].abs() > t['model_err'].abs() + 0.05, 'no', '='))
print(t[['mat', 'sub', 'adj', 'model_err', 'new_err', 'moved_toward_model']].to_string(index=False))
n_to = (t['moved_toward_model'] == 'YES').sum()
n_eq = (t['moved_toward_model'] == '=').sum()
print(f"\n{n_to} of {len(t)} bonds moved TOWARD the model, {n_eq} unchanged")
