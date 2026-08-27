"""
calibration -- the single source for self-calibrated constants.

Why this exists (8/27): hardcoded copies of the concession drifted across
tools -- rank_deals ran 7 bps out of sync with the pipeline. Nothing outside
this module may hardcode a concession, its sigma, or the trusted-error scale.

Concession definition (unified 8/27, fix #5): measured on tranches with
>= MIN_WORKOUT_YRS to their WORKOUT date (call date if priced-to-call, else
maturity) -- the same axis price_wire's tenor ramp uses. Previously the
tracker used priced-to-call rows while the ramp used calendar maturity;
the two definitions diverged on non-PTC long bonds.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MIN_WORKOUT_YRS = 8
FALLBACK_CONCESSION = 10.0
FALLBACK_SIGMA = 3.0
FALLBACK_TRUSTED_P90 = 8.6


def _latest_per_deal():
    latest = {}
    for f in (HERE / 'wire_archive').glob('*.csv'):
        tag = re.sub(r'_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', '', f.stem)
        if tag not in latest or f.stat().st_mtime > latest[tag].stat().st_mtime:
            latest[tag] = f
    return latest


def _implied_per_deal():
    vals = {}
    for tag, f in _latest_per_deal().items():
        try:
            d = pd.read_csv(f)
            if 'Error (bps)' not in d.columns or 'Concession Used (bps)' not in d.columns or not len(d):
                continue
            used = float(d['Concession Used (bps)'].iloc[0])
            ref = datetime.fromtimestamp(f.stat().st_mtime)
            workout = d.apply(
                lambda r: r['Maturity'] if r.get('Priced To') == 'Maturity' else r.get('Priced To'),
                axis=1)
            yrs = pd.to_datetime(workout, format='%m/%d/%Y', errors='coerce').map(
                lambda t: (t - ref).days / 365 if pd.notna(t) else float('nan'))
            long_rows = d[yrs >= MIN_WORKOUT_YRS]
            if len(long_rows):
                vals[tag] = used - long_rows['Error (bps)'].mean()
        except Exception:
            continue
    return vals


def concession(verbose=True):
    vals = _implied_per_deal()
    if not vals:
        if verbose:
            print(f"concession default: {FALLBACK_CONCESSION} bps (no archive)")
        return FALLBACK_CONCESSION
    c = round(sum(vals.values()) / len(vals), 1)
    if verbose:
        print(f"concession auto-calibrated: {c} bps from {len(vals)} archived deal(s)")
    return c


def concession_sigma():
    vals = list(_implied_per_deal().values())
    if len(vals) < 3:
        return FALLBACK_SIGMA
    s = pd.Series(vals).std(ddof=1)
    return round(max(float(s), 2.0), 1)   # floor: never overstate certainty


def trusted_p90():
    p = HERE / 'trust_stats.json'
    if p.exists():
        try:
            return float(json.loads(p.read_text())['trusted_p90_bps'])
        except Exception:
            pass
    return FALLBACK_TRUSTED_P90


def write_trust_stats(stats: dict):
    stats = dict(stats)
    stats['written'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    (HERE / 'trust_stats.json').write_text(json.dumps(stats, indent=1))
    print(f"trust_stats.json updated: {stats}")
