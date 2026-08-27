"""
spline_features -- Spline Data integration per the desk spec (8/26):
use it THROUGH DV01, and stick to 1mm+ / _lrg_lqd only.

Spline's curves are indexed by DURATION (DV01 bucket 1..30), not calendar
tenor -- matching the original analyze_pricing lookup. That makes the curve
feature NEW-ISSUE-SAFE: a wire tranche's DV01 is computable analytically,
so its curve level exists before the bond does.

Provides:
    load_curves()        -- _lrg_lqd curves as a (curve_type, dv01_bucket) lookup
    load_pricing_1mm()   -- per-CUSIP 1mm mid yield, empirical DV01, spread
    attach_curve_yield(df) -- adds 'spline_curve_yield' column to any frame
                              that has composite_rating, muni_security_type_desc,
                              and par_dv01 (training bonds and wire tranches alike)

Requires spline_curves.csv / spline_pricing.csv from spline_pull.py.
All functions fail LOUDLY if the data is missing -- no silent NaN columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# composite_rating (1-22) -> Spline's four rating families (their scheme)
RATING_FAMILY = {1: 'aaa', 2: 'aa', 3: 'aa', 4: 'aa', 5: 'a', 6: 'a', 7: 'a',
                 8: 'bbb', 9: 'bbb', 10: 'bbb'}


def load_curves():
    p = HERE / 'spline_curves.csv'
    if not p.exists():
        raise FileNotFoundError(
            "spline_curves.csv not found -- run spline_pull.py first "
            "(needs SPLINE_PASSWORD; snapshots exist during market hours).")
    c = pd.read_csv(p)
    c = c[c['curve'].str.endswith('_lrg_lqd')]          # 1mm+ liquidity only
    c['tenor'] = pd.to_numeric(c['tenor'], errors='coerce')
    return c.set_index([c['curve'], 'tenor'])['value']


def load_pricing_1mm():
    p = HERE / 'spline_pricing.csv'
    if not p.exists():
        raise FileNotFoundError("spline_pricing.csv not found -- run spline_pull.py first.")
    s = pd.read_csv(p)
    s = s[pd.to_numeric(s['par_traded_bucket'], errors='coerce') == 1_000_000]
    out = pd.DataFrame(index=s['cusip'])
    by, ay = (pd.to_numeric(s[c], errors='coerce').to_numpy()
              for c in ('bid_yield', 'ask_yield'))
    bp, ap = (pd.to_numeric(s[c], errors='coerce').to_numpy()
              for c in ('bid_price', 'ask_price'))
    out['spline_mid_yield'] = (by + ay) / 2
    with np.errstate(all='ignore'):
        out['spline_dv01'] = (ap - bp) / (by - ay)      # empirical, 1mm size
    out['spline_spread_bps'] = (by - ay) * 100
    return out[~out.index.duplicated()]


def curve_type(rating_num, security_type_desc):
    fam = RATING_FAMILY.get(int(rating_num) if pd.notna(rating_num) else -1)
    if fam is None:
        return None
    is_go = bool(pd.Series([str(security_type_desc)]).str
                 .contains('G.O|Double barreled', case=False, regex=True).iloc[0])
    return fam + ('go' if is_go else 'rev') + '_lrg_lqd'


def attach_curve_yield(df, curves=None):
    """Add 'spline_curve_yield': Spline's live curve level for each bond's
    rating-family x go/rev bucket at its DV01 duration bucket. Works on any
    frame with composite_rating, muni_security_type_desc, par_dv01 -- which
    includes wire-tranche frames (par_dv01 is analytic)."""
    curves = curves if curves is not None else load_curves()
    df = df.copy()
    rating = pd.to_numeric(df.get('composite_rating'), errors='coerce')
    dv01_bucket = np.ceil(pd.to_numeric(df.get('par_dv01'), errors='coerce')).clip(1, 30)
    keys = [
        (curve_type(r, t), b) if pd.notna(r) and pd.notna(b) else (None, np.nan)
        for r, t, b in zip(rating, df.get('muni_security_type_desc'), dv01_bucket)
    ]
    idx = pd.MultiIndex.from_tuples(keys)
    df['spline_curve_yield'] = curves.reindex(idx).to_numpy()
    n = df['spline_curve_yield'].notna().sum()
    print(f"spline_curve_yield attached: {n:,}/{len(df):,} rows matched a curve")
    return df
