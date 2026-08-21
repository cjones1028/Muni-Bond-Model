"""
concession_tracker -- learn the new-issue concession from the deal archive.

    python concession_tracker.py

Each archived deal's implied concession = concession used at run time minus
the deal's mean signed error (if the model came out 3 bps below the wire on
average with 13 applied, the deal's true concession was ~16). Reports the
per-deal history and the recommended estimate + spread for rank_deals.py's
SIGMA_CONC. Gets sharper with every wire priced -- this is the cheapest
accuracy gain in the system.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARCH = HERE / 'wire_archive'

# one vote per DEAL: re-priced wires create multiple files; use each deal's
# newest archive only
import re as _re
latest = {}
for f in ARCH.glob('*.csv'):
    tag = _re.sub(r'_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', '', f.stem)
    if tag not in latest or f.stat().st_mtime > latest[tag].stat().st_mtime:
        latest[tag] = f

rows = []
for f in sorted(latest.values()):
    d = pd.read_csv(f)
    if 'Error (bps)' not in d.columns or not len(d):
        continue
    used = (float(d['Concession Used (bps)'].iloc[0])
            if 'Concession Used (bps)' in d.columns else 13.0)  # pre-tracking runs
    implied = used - d['Error (bps)'].mean()
    rows.append({'deal': f.stem[:44], 'tranches': len(d),
                 'concession used': used,
                 'implied concession (bps)': round(implied, 1)})

if not rows:
    sys.exit("no archived deals found in wire_archive\\")

t = pd.DataFrame(rows)
print(t.to_string(index=False))

imp = t['implied concession (bps)']
est, spread = imp.mean(), imp.std(ddof=1) if len(imp) > 1 else 3.0
print(f"\nacross {len(t)} deal(s): recommended concession {est:.0f} bps"
      f" | spread {spread:.1f} bps")
print(f"use:  run_pipeline.py --concession {est:.0f}")
if len(t) >= 5:
    print("5+ deals: consider splitting the estimate by rating tier "
          "(IG vs BBB priced differently in the first two already).")
else:
    print(f"({len(t)} deals is a thin estimate -- it tightens with every wire priced)")
