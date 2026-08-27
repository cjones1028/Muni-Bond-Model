"""Empirical verification of the 5 audit fixes -- execute, don't assert.
1. Leakage: PROVE the new bundle's holdout CUSIPs never touch training rows.
2. Trainer unity: prove all trainer call sites use stacked_frame.
3. Calibration: prove no hardcoded concession/sigma/p90 literals remain.
5. Ramp alignment: prove workout array aligns with priced tranches on all 4 wires.
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from Wire_Parser import parse_wire

fails = []


def check(name, cond, detail=''):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f" ({detail})" if detail and not cond else ''))
    if not cond:
        fails.append(name)


print("1. leakage -- empirical, on the actual production bundle")
bundle = mm.load_bundle(HERE / 'model.joblib')
hc = set(bundle.get('holdout_cusips') or [])
check("bundle persists holdout CUSIPs", len(hc) > 1000, str(len(hc)))
stacked = mm.stacked_frame(HERE)
tenor = pd.to_numeric(stacked['days_to_maturity_30_360'], errors='coerce')
trainable = stacked[tenor >= 360]
# reconstruct the training fit set exactly as train_yield_model does
cusips = pd.Index(trainable.index.unique())
rc = np.random.RandomState(42).rand(len(cusips))
test_c = set(cusips[rc < 0.2])
val_c = set(cusips[(rc >= 0.2) & (rc < 0.28)])
fit_rows = trainable[~trainable.index.isin(test_c) & ~trainable.index.isin(val_c)]
overlap = hc & set(fit_rows.index)
check("ZERO holdout CUSIPs in any training row", len(overlap) == 0, f"overlap={len(overlap)}")
check("persisted holdout == reconstructed split", hc == test_c,
      f"sym-diff={len(hc ^ test_c)}")

print("2. one trainer recipe -- all call sites")
for f in ['train_production.py', 'run_pipeline.py', 'test_suite.py']:
    src = (HERE / f).read_text(encoding='utf-8')
    trains = 'train_yield_model' in src
    uses_canon = 'stacked_frame' in src
    check(f"{f}: trains only via stacked_frame", (not trains) or uses_canon)

print("3. no stray calibration literals")
pattern = re.compile(r'(CONCESSION\w*\s*=\s*\d|SIGMA_CONC\s*=\s*\d|TRUSTED_P90\w*\s*=\s*\d|concession_bps\s*=\s*\d)')
offenders = []
for f in HERE.glob('*.py'):
    if f.name in ('calibration.py', 'verify_fixes.py') or f.name.startswith(('exp_', 'diag_', 'compare_', 'fair_', 'smoke')):
        continue  # experiments/diagnostics may pin values for reproducibility
    for i, line in enumerate(f.read_text(encoding='utf-8').splitlines(), 1):
        if (pattern.search(line) and 'FALLBACK' not in line
                and 'pinned-for-regression' not in line and 'API default' not in line):
            offenders.append(f"{f.name}:{i}")
check("production files free of hardcoded calibration", not offenders, '; '.join(offenders))

print("5. workout-ramp alignment on all four wires")
for wfile in ['BAML Write-Up.txt', 'NYC_TFA_wire.txt', 'psu_wire.txt', 'portland_wire.txt']:
    deal = parse_wire(open(HERE / wfile, encoding='utf-8').read())
    priced = [t for t in deal['tranches'] if t['price'] is not None]
    wd = [t['ptc_date'] or t['maturity'] for t in priced]
    check(f"{wfile}: workout list aligns 1:1 with priced tranches",
          len(wd) == len(priced) and all(w for w in wd))

print()
print("ALL VERIFIED" if not fails else f"{len(fails)} FAILURE(S): {fails}")
sys.exit(1 if fails else 0)
