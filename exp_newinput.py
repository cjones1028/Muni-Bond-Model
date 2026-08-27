"""The 'new input' certainty test, two parts:

A. LEAVE-ONE-OUT deals: price each deal with the concession calibrated from
   the OTHER three only -- a new deal's actual situation.
B. PSEUDO-WIRES from unseen bonds: construct wire-format 'deals' from real
   ICE-marked bonds in the model's persisted holdout (issuers never priced,
   bonds never trained on), run the FULL pipeline (parse -> auto-issuer ->
   template -> model), concession 0 (secondary marks carry none).
   Restricted to non-callable, non-refunded, >=1.5yr bonds so the missing
   CALL FEATURES block is truthful.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from calibration import _implied_per_deal
from Wire_Parser import parse_wire

PY = sys.executable

print("=== A. leave-one-out concession (a new deal's situation) ===")
implied = _implied_per_deal()
tdf = pd.read_parquet(HERE / 'template_cache.parquet')
bundle = mm.load_bundle(HERE / 'model.joblib')
DEALS = [('LA', 'BAML Write-Up.txt', 'LOS ANG', 'LOS_ANGELES'),
         ('NYC', 'NYC_TFA_wire.txt', 'CITY TRANSITIONAL FIN', 'Preliminary_Pricing'),
         ('PSU', 'psu_wire.txt', 'PENNSYLVANIA STATE UNIVERSITY', 'PENNSYLVANIA'),
         ('PDX', 'portland_wire.txt', 'PORTLAND ORE SWR', 'PORTLAND')]
loo_means = []
for name, wfile, iss, tagkey in DEALS:
    others = [v for t, v in implied.items() if tagkey not in t]
    c_loo = round(sum(others) / len(others), 1)
    deal = parse_wire(open(HERE / wfile, encoding='utf-8').read())
    templ = mm.template_from(tdf, issuer_contains=iss)
    r = mm.price_wire(deal, bundle, templ, concession_bps=c_loo)
    loo_means.append(r['Error (bps)'].abs().mean())
    print(f">>> {name}: LOO concession {c_loo} | mean abs {loo_means[-1]:.1f} "
          f"| median {r['Error (bps)'].abs().median():.1f}")
print(f"LOO expected-new-deal error: {np.mean(loo_means):.1f} bps mean-of-means")

print("\n=== B. pseudo-wires from provably unseen bonds ===")
df = mm.clean_universe(mm.load_evals(HERE / 'ICE_Evals.csv'))
hold = df[df.index.isin(set(bundle['holdout_cusips']))]
cand = hold[
    (hold['call_indicator'].astype(str).str.lower() != 'true')
    & hold['called_redemption_type_desc'].isna()
    & (pd.to_numeric(hold['days_to_maturity_30_360'], errors='coerce') > 540)
    & hold['primary_name_abbreviated'].notna()
]
used = ('LOS ANG', 'TRANSITIONAL', 'PENNSYLVANIA STATE', 'PORTLAND')
groups = cand.groupby('primary_name_abbreviated')
sizes = groups.size().sort_values(ascending=False)
picked = [n for n in sizes.index
          if not any(u in str(n).upper() for u in used)][:3]

today = pd.Timestamp.now()
for issuer_name in picked:
    g = groups.get_group(issuer_name).copy()
    g['mat_dt'] = pd.to_datetime(g['maturity_date'], errors='coerce')
    g = g.dropna(subset=['mat_dt', 'target_yield', 'mid']).sort_values('mat_dt').head(12)
    if len(g) < 5:
        continue
    lines = [f"RE: $ {int(pd.to_numeric(g['outstanding_amount']).sum()):,}*",
             str(issuer_name), 'PSEUDO-WIRE TEST (real ICE marks as yields)', '',
             f"DATED:{today:%m/%d/%Y}", '']
    for _, b in g.iterrows():
        amt = f"{max(int(pd.to_numeric(b['outstanding_amount']) / 1000), 1):,}M"
        lines.append(f"{b['mat_dt']:%m/%d/%Y}     {amt:>10}     "
                     f"{float(b['current_coupon_rate']):.2f}%     {float(b['target_yield']):.2f}")
        lines.append(f"                      (Approx. $ Price {float(b['mid']):.3f})")
    (HERE / 'pseudo_wire.txt').write_text('\n'.join(lines), encoding='utf-8')
    out = subprocess.run([PY, str(HERE / 'run_pipeline.py'), '--wire', 'pseudo_wire.txt',
                          '--concession', '0', '--no-archive'],
                         capture_output=True, text=True, cwd=HERE)
    for ln in out.stdout.splitlines():
        if 'auto-matched' in ln or 'mean abs' in ln or 'credit mismatch' in ln:
            print(f"  [{str(issuer_name)[:34]}] {ln.strip()}")
