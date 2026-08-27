"""
screen_universe -- rank the whole universe by model-vs-market gap, separated
by how much the model's number can be trusted on each bond.

    python screen_universe.py [--min-gap 10] [--min-size 5000000] [--top 15]

For every bond: model fair yield (3-model ensemble) vs ICE mid yield.
  gap > 0  -> market yield ABOVE model fair value -> screens CHEAP
  gap < 0  -> market yield below model            -> screens RICH

Tiers (confidence about the model's own number, computed in advance):
  A  no trust flags, ensemble std <= 2 bps  -> model's cleanest territory
  B  no trust flags, ensemble std <= 5 bps  -> still trusted
  C  any trust flag                         -> RESEARCH ONLY: the market may
     know something the model can't see (distress, escrow, illiquidity).
     A gap here is a lead for credit work, never a signal.

Edge score = gap / p90 of the trusted universe's holdout error -- i.e. "how
many typical model errors wide is this gap". Below ~1.5 the gap is
indistinguishable from model noise.

Est. $ / $1mm face = gap x DV01: what convergence would be worth. ICE mids
are marks, not executable prices -- muni bid/ask costs are real; the
--cost-bps hurdle (default 10) nets an assumed round trip out of the score.

Output: screen_actionable.csv (tiers A/B) and screen_research.csv (tier C),
top rows of each printed. Decision support for a professional -- position
decisions stay human.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm

# scale unit for edge_score: holdout p90 error of the TRUSTED set, from the
# calibration source (diag_confidence.py refreshes it after retrains)
from calibration import trusted_p90
TRUSTED_P90_BPS = trusted_p90()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-gap', type=float, default=10.0, help='min |gap| bps to list')
    ap.add_argument('--min-size', type=float, default=5_000_000, help='min outstanding $')
    ap.add_argument('--cost-bps', type=float, default=10.0, help='assumed round-trip cost, bps')
    ap.add_argument('--top', type=int, default=15)
    args = ap.parse_args()

    print("loading evals + model...")
    df = mm.load_evals(HERE / 'ICE_Evals.csv')
    df = mm.clean_universe(df)
    dfF = mm.build_features(df)
    bundle = mm.load_bundle(HERE / 'model.joblib')

    X = mm._prep_matrix(dfF, bundle['numeric'], bundle['categorical'], bundle['categories'])
    members = np.array([m.predict(X) for m in bundle['models']])
    dfF['model_yield'] = members.mean(axis=0)
    dfF['ens_std_bps'] = members.std(axis=0) * 100
    dfF['gap_bps'] = (dfF['target_yield'] - dfF['model_yield']) * 100

    flags, trusted = mm.confidence_flags(dfF, dfF['model_yield'], dfF['ens_std_bps'])
    dfF['tier'] = np.where(~trusted, 'C',
                           np.where(dfF['ens_std_bps'] <= 2, 'A', 'B'))
    dfF['flags'] = flags.apply(lambda r: ','.join(flags.columns[r.values]), axis=1)

    # a gap far beyond the model's own error distribution is not an
    # opportunity -- it is evidence of something the model can't see (bad
    # mark, escrow, undisclosed story). Auto-demote to research.
    implausible = (dfF['tier'] != 'C') & (dfF['gap_bps'].abs() > 50)
    dfF.loc[implausible, 'tier'] = 'C'
    dfF.loc[implausible, 'flags'] = (dfF.loc[implausible, 'flags'] + ',gap_implausible').str.lstrip(',')

    # capturability: cost each bond by its OWN quoted bid/ask width (in yield
    # bps), floored at --cost-bps. An illiquid bond's fat spread eats its gap.
    quoted_spread = ((pd.to_numeric(dfF.get('yield_bid'), errors='coerce')
                      - pd.to_numeric(dfF.get('yield_offer'), errors='coerce')) * 100)
    dfF['cost_bps'] = quoted_spread.clip(lower=args.cost_bps).fillna(args.cost_bps).round(1)
    dfF['edge_score'] = ((dfF['gap_bps'].abs() - dfF['cost_bps']).clip(lower=0)
                         / TRUSTED_P90_BPS).round(2)
    dv01 = pd.to_numeric(dfF.get('ice_dv01'), errors='coerce').clip(lower=0, upper=30)
    dfF['est_$_per_mm'] = (dfF['gap_bps'].abs() / 100 * dv01 * 10_000).round(0)

    size_ok = pd.to_numeric(dfF['outstanding_amount'], errors='coerce') >= args.min_size
    big_gap = dfF['gap_bps'].abs() >= args.min_gap
    view_cols = ['tier', 'gap_bps', 'cost_bps', 'edge_score', 'est_$_per_mm', 'ens_std_bps',
                 'target_yield', 'model_yield', 'composite_rating',
                 'current_coupon_rate', 'days_to_maturity_30_360',
                 'outstanding_amount', 'purpose_class_desc',
                 'incorporated_state_code_desc', 'flags']
    view_cols = [c for c in view_cols if c in dfF]

    act = (dfF[size_ok & big_gap & (dfF['tier'] != 'C')]
           .sort_values('edge_score', ascending=False))
    res = (dfF[size_ok & big_gap & (dfF['tier'] == 'C')]
           .sort_values('edge_score', ascending=False))

    act[view_cols].to_csv(HERE / 'screen_actionable.csv')
    res[view_cols].to_csv(HERE / 'screen_research.csv')

    print(f"\nuniverse {len(dfF):,} | tier A {(dfF['tier']=='A').sum():,} "
          f"| tier B {(dfF['tier']=='B').sum():,} | tier C {(dfF['tier']=='C').sum():,}")
    print(f"screens (|gap|>={args.min_gap}bps, size>={args.min_size:,.0f}): "
          f"actionable {len(act):,} -> screen_actionable.csv | "
          f"research {len(res):,} -> screen_research.csv")

    show = ['tier', 'gap_bps', 'cost_bps', 'edge_score', 'est_$_per_mm', 'ens_std_bps',
            'target_yield', 'model_yield', 'composite_rating', 'outstanding_amount']
    print(f"\n=== top {args.top} ACTIONABLE (tiers A/B, ranked by edge score) ===")
    print(act[show].head(args.top).round(2).to_string())
    print(f"\n=== top {args.top} RESEARCH LEADS (tier C: verify the story first) ===")
    print(res[show + ['flags']].head(args.top).round(2).to_string())


if __name__ == '__main__':
    main()
