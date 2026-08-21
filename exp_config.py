"""
exp_config -- pick the model configuration on evidence: for each candidate,
train once (single model, fixed split) and score BOTH the universe holdout
and the two real wires. Decides the final production config.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from Wire_Parser import parse_wire

df = mm.load_evals(HERE / 'ICE_Evals.csv')
df = mm.clean_universe(df)
baml = parse_wire(open(HERE / 'BAML Write-Up.txt', encoding='utf-8').read())
nyc = parse_wire(open(HERE / 'NYC_TFA_wire.txt', encoding='utf-8').read())

OLD_NUM = [c for c in mm.NUMERIC_FEATURES
           if c not in ('issue_price', 'min_denom_amount', 'denom_increment_amount',
                        'bond_age_days', 'days_to_next_coupon')]
NEW_NO_AGE = [c for c in mm.NUMERIC_FEATURES if c != 'bond_age_days']

CONFIGS = [
    ('A l2 / old feats (orig)',  dict(numeric=OLD_NUM, n_ensemble=1,
                                      objective='l2', n_estimators=1200)),
    ('B l1 / old feats',         dict(numeric=OLD_NUM, n_ensemble=1)),
    ('C l1 / new feats no age',  dict(numeric=NEW_NO_AGE, n_ensemble=1)),
    ('D l1 / new feats + age',   dict(numeric=list(mm.NUMERIC_FEATURES), n_ensemble=1)),
]

t_la = mm.template_from(df, issuer_contains='LOS ANG')
t_ny = mm.template_from(df, issuer_contains='CITY TRANSITIONAL FIN')

for name, kw in CONFIGS:
    b = mm.train_yield_model(df, seed=42, **kw)
    r_la = mm.price_wire(baml, b, t_la, concession_bps=13)
    r_ny = mm.price_wire(nyc, b, t_ny, concession_bps=13)
    la = r_la['Error (bps)']
    ny = r_ny['Error (bps)']
    print(f">>> {name}: holdout {b['mae_bps']:.1f}/{b['median_bps']:.1f} | "
          f"LA {la.abs().mean():.1f} (signed {la.mean():+.1f}) | "
          f"NYC {ny.abs().mean():.1f} (signed {ny.mean():+.1f})")
