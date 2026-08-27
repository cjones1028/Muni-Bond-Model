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

# implied concessions from the single calibration source (workout-axis
# definition, one vote per deal -- see calibration.py)
from calibration import _implied_per_deal, _latest_per_deal

implied = _implied_per_deal()
files = _latest_per_deal()
rows = []
for tag, val in sorted(implied.items()):
    d = pd.read_csv(files[tag])
    used = (float(d['Concession Used (bps)'].iloc[0])
            if 'Concession Used (bps)' in d.columns else float('nan'))
    rows.append({'deal': tag[:44], 'tranches': len(d),
                 'concession used': used,
                 'implied concession (bps)': round(val, 1)})

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
