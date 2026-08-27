"""Three experiments enabled by having two dated snapshots (8/19, 8/25):

A. TIME-FORWARD validation (the honest test, first time possible):
   train on 8/19 marks -> predict 8/25 marks. Measures real 6-day-forward
   accuracy, not same-day replication.
B. STACKED training: train on both snapshots (holdout CUSIPs excluded from
   BOTH) -> does temporal variety help? Judged on 8/25 holdout + wires.
C. CAPACITY probe: deeper trees (num_leaves 255, min_child 15) on the
   stacked data, same judging.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from Wire_Parser import parse_wire

old = mm.clean_universe(mm.load_evals(HERE / 'evals_archive/ICE_Evals_2026-08-19.csv'))
new = mm.clean_universe(mm.load_evals(HERE / 'evals_archive/ICE_Evals_2026-08-25.csv'))

# holdout: seed-42 20% of the NEW snapshot, by CUSIP, excluded from all training
rng = np.random.RandomState(42)
hold_cusips = set(new.index[rng.rand(len(new)) < 0.2])
new_hold = new[new.index.isin(hold_cusips)]
newF_hold = mm.build_features(new_hold)

WIRES = [('LA', 'BAML Write-Up.txt', 'LOS ANG'),
         ('NYC', 'NYC_TFA_wire.txt', 'CITY TRANSITIONAL FIN'),
         ('PSU', 'psu_wire.txt', 'PENNSYLVANIA STATE UNIVERSITY'),
         ('PDX', 'portland_wire.txt', 'PORTLAND ORE SWR')]


def grade(b, label):
    X = mm._prep_matrix(newF_hold, b['numeric'], b['categorical'], b['categories'])
    err = np.abs(mm.predict_bundle(b, X) - newF_hold['target_yield'].to_numpy()) * 100
    parts = []
    for wname, wfile, iss in WIRES:
        deal = parse_wire(open(HERE / wfile, encoding='utf-8').read())
        templ = mm.template_from(new, issuer_contains=iss)
        r = mm.price_wire(deal, b, templ, concession_bps=7.0)
        parts.append(f"{wname} {r['Error (bps)'].abs().mean():.1f}")
    print(f">>> {label}: 8/25-holdout MAE {err.mean():.2f} | median {np.median(err):.2f} "
          f"| wires: {' | '.join(parts)}")


# A. time-forward: trained on 8/19 ONLY (holdout CUSIPs removed), graded on 8/25
b_fwd = mm.train_yield_model(old[~old.index.isin(hold_cusips)], n_ensemble=1, seed=42)
grade(b_fwd, "A TRAIN 8/19 -> PREDICT 8/25 (time-forward)")

# same-day reference: trained on 8/25 (holdout removed), graded on same 8/25 slice
b_same = mm.train_yield_model(new[~new.index.isin(hold_cusips)], n_ensemble=1, seed=42)
grade(b_same, "REF train 8/25 -> predict 8/25 (same-day)")

# B. stacked: both snapshots, holdout CUSIPs removed from both
stacked = pd.concat([old[~old.index.isin(hold_cusips)],
                     new[~new.index.isin(hold_cusips)]])
b_stack = mm.train_yield_model(stacked, n_ensemble=1, seed=42)
grade(b_stack, "B STACKED 8/19+8/25")

# C. deeper trees on stacked
b_deep = mm.train_yield_model(stacked, n_ensemble=1, seed=42,
                              num_leaves=255, min_child_samples=15)
grade(b_deep, "C STACKED + deeper trees")
