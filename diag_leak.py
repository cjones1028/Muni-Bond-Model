"""Quantify suspected holdout leakage under stacked training:
(a) inside train_yield_model: row-split means a CUSIP held out from its 8/25
    row can still be TRAINED ON via its 8/19 row;
(b) in post-hoc evaluations (diag_confidence, bucket_errors): their seed-42
    mask over the single snapshot does not match the training mask over the
    stacked frame, so 'holdout' bonds there may have been trained on."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm

old = mm.clean_universe(mm.load_evals(HERE / 'evals_archive/ICE_Evals_2026-08-19.csv'))
new = mm.clean_universe(mm.load_evals(HERE / 'evals_archive/ICE_Evals_2026-08-25.csv'))
stacked = pd.concat([old, new])

# reproduce train_yield_model's split on the stacked frame (before its
# internal sub-1yr drop, which uses the same row order -- approximation is
# fine for measuring overlap magnitude)
rng = np.random.RandomState(42)
r = rng.rand(len(stacked))
test_rows = stacked.index[r < 0.2]
train_rows = stacked.index[~(r < 0.2) & ~((r >= 0.2) & (r < 0.28))]

test_cusips = set(test_rows)
train_cusips = set(train_rows)
both = test_cusips & train_cusips
print(f"(a) train_yield_model internal split: {len(test_cusips):,} held-out CUSIPs, "
      f"{len(both):,} ({100*len(both)/len(test_cusips):.0f}%) ALSO appear in training "
      f"via their other snapshot -> internal holdout contaminated")

# post-hoc mask (diag_confidence / rank_deals style): seed-42 over the single
# 8/25 snapshot
rng2 = np.random.RandomState(42)
posthoc_hold = set(new.index[rng2.rand(len(new)) < 0.2])
leaked = posthoc_hold & train_cusips
print(f"(b) post-hoc evaluations: {len(posthoc_hold):,} 'holdout' CUSIPs, "
      f"{len(leaked):,} ({100*len(leaked)/len(posthoc_hold):.0f}%) were in the stacked "
      f"training set -> diag_confidence/bucket_errors stats contaminated")
