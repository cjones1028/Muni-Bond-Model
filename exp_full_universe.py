"""Can the full 1.24M-bond universe (exposed by ICE's 8/31 re-shard) help
wire pricing if filtered to institutional size?

Facts driving this experiment (measured 8/31):
  - old universe = ICE's institutional slice: median $11.1mm outstanding
  - new-only 1.12M bonds: median $425k -- retail odd-lots, matrix marks
  - naive full-universe training degraded the pinned wire regressions
    LA 5.6 -> 9.0, NYC 4.4 -> 8.9 (test_suite, concession pinned 13)

Design: stack the two known-good snapshots as-is, add the 8/31 full snapshot
filtered to outstanding >= THRESH, train (seed 42, same recipe), and score
the two pinned regression wires. Adopt only if both beat/match the restored
baseline. Models saved to exp_model_<tag>.joblib -- production model.joblib
is NOT touched.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm  # noqa: E402
from Wire_Parser import parse_wire  # noqa: E402

FULL_SNAP = HERE / 'evals_archive' / 'ICE_Evals_2026-08-31.csv.full'

print("loading snapshots...")
old_frames = []
for f in [HERE / 'evals_archive' / 'ICE_Evals_2026-08-19.csv',
          HERE / 'evals_archive' / 'ICE_Evals_2026-08-25.csv']:
    old_frames.append(mm.clean_universe(mm.load_evals(f)))
full = mm.clean_universe(mm.load_evals(FULL_SNAP))
amt = pd.to_numeric(full['outstanding_amount'], errors='coerce')

baml = parse_wire(open(HERE / 'BAML Write-Up.txt', encoding='utf-8').read())
nyc = parse_wire(open(HERE / 'NYC_TFA_wire.txt', encoding='utf-8').read())
tmpl_df = old_frames[-1]
templ_la = mm.template_from(tmpl_df, issuer_contains='LOS ANG')
templ_ny = mm.template_from(tmpl_df, issuer_contains='CITY TRANSITIONAL FIN')

RESULTS = []
for thresh, tag in [(5_000_000, '5mm'), (1_000_000, '1mm')]:
    extra = full[amt >= thresh]
    stack = pd.concat(old_frames + [extra])
    print(f"\n=== THRESH {tag}: 8/31 slice {len(extra):,} rows, stack {len(stack):,} ===")
    bundle = mm.train_yield_model(stack, seed=42)
    mm.save_bundle(bundle, HERE / f'exp_model_{tag}.joblib')
    row = {'config': tag, 'stack_rows': len(stack), 'holdout_mae': bundle['mae_bps']}
    for wname, deal, templ in [('LA', baml, templ_la), ('NYC', nyc, templ_ny)]:
        res = mm.price_wire(deal, bundle, templ, concession_bps=13)  # pinned-for-regression
        row[wname] = round(res['Error (bps)'].abs().mean(), 1)
    RESULTS.append(row)
    print(row)

print("\n=== SUMMARY (baseline restored model: LA 5.6 / NYC 4.4, pinned conc 13) ===")
print(pd.DataFrame(RESULTS).to_string(index=False))
print("\nDecision rule: adopt a config ONLY if LA and NYC both <= baseline.")
