"""
smoke_test -- generate a synthetic ICE evals CSV with the real column schema
and drive the full pipeline (train -> parse wire -> price) end to end.

    python smoke_test.py

This proves the plumbing works before real data exists. The synthetic yields
follow a plausible curve + rating/coupon/call effects, so the model should
learn them to within a few bps -- if it does, the harness is wired correctly.
Numbers mean nothing beyond that.
"""

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
rng = np.random.RandomState(7)
N = 20000

STATES = ['California', 'New York', 'Texas', 'Florida', 'Illinois']
SECTORS = ['Transportation', 'Education', 'Utilities', 'Health care', 'General purpose']
SUBSECTORS = ['Toll road', 'Airport', 'Highway', 'School district', 'Water & sewer']
MUNI_TYPES = ['Revenue', 'G.O.', 'Double barreled']
RATING_NUM = rng.choice(range(1, 13), N, p=np.array([6,7,10,10,12,12,10,9,8,7,5,4])/100)

tenor_yrs = rng.uniform(0.5, 32, N)
coupon = rng.choice([3.0, 4.0, 5.0, 5.25, 5.5], N, p=[.1, .2, .55, .1, .05])
callable_ = tenor_yrs > rng.uniform(8, 12, N)
call_yrs = np.where(callable_, np.clip(tenor_yrs - rng.uniform(5, 15, N), 1, None), np.nan)

# ground truth: curve + rating spread + call premium + coupon effect + noise
curve = 2.6 + 1.05 * np.log1p(tenor_yrs) - 0.15 * np.log1p(tenor_yrs) ** 2 + 0.055 * tenor_yrs
y = (curve + 0.11 * (RATING_NUM - 1) + 0.18 * callable_.astype(float)
     - 0.05 * (coupon - 5.0) + rng.normal(0, 0.02, N))

dur = np.minimum(tenor_yrs, np.where(callable_, np.nan_to_num(call_yrs, nan=99), tenor_yrs)) * 0.8
mid = 100 + (coupon - y) * dur + rng.normal(0, 0.15, N)

def rand_cusips(n):
    chars = np.array(list('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'))
    return [''.join(rng.choice(chars, 9)) for _ in range(n)]

issue = np.exp(rng.uniform(15, 21, N))
df = pd.DataFrame({
    'yield_bid': y + 0.03, 'yield_offer': y - 0.03,
    'bid': mid - 0.15, 'offer': mid + 0.15, 'mid': mid,
    'current_coupon_rate': coupon,
    'days_to_maturity_30_360': (tenor_yrs * 360).round(),
    'days_to_next_call_30_360': np.where(callable_, (call_yrs * 360).round(), np.nan),
    'days_to_first_call_30_360': np.where(callable_, (call_yrs * 360).round(), np.nan),
    'days_from_call_to_maturity_30_360': np.where(callable_, ((tenor_yrs - call_yrs) * 360).round(), np.nan),
    'days_to_refund_30_360': np.nan, 'days_to_next_sink_date_30_360': np.nan,
    'next_sink_price': np.nan, 'principal_factor': 1.0,
    'issue_amount': issue,
    'outstanding_amount': issue * rng.uniform(0.02, 0.15, N),
    'next_call_price': np.where(callable_, 100.0, np.nan),
    'call_indicator': callable_.astype(str),
    'federal_tax_status_desc': 'Tax-exempt',
    'coupon_type_desc': 'Fixed rate',
    'default_indicator': False,
    'incorporated_state_code_desc': rng.choice(STATES, N),
    'muni_security_type_desc': rng.choice(MUNI_TYPES, N, p=[.6, .3, .1]),
    'composite_rating': RATING_NUM,
    'normalized_moody_long_rating': RATING_NUM.astype(str),
    'normalized_sandp_long_rating': RATING_NUM.astype(str),
    'normalized_moody_enhanced_long_rating': RATING_NUM.astype(str),
    'enhanced_composite_rating': RATING_NUM,
    'purpose_class_desc': rng.choice(SECTORS, N),
    'purpose_sub_class_desc': rng.choice(SUBSECTORS, N),
    'use_of_proceeds_desc': rng.choice(SUBSECTORS, N),
    'security_class': 'Revenue bond',
    'state_tax_status_desc': 'Exempt',
    'ad_valorem_tax_status_desc': 'No',
    'bond_insurance': 'None',
    'bank_qualified': 'N',
    'call_notice': '30',
    'distinct_call_timing_desc': np.where(callable_, 'Continuously callable', 'Non-callable'),
    'organization_master_id': rng.choice([f'ORG{i:04d}' for i in range(400)], N),
    'conduit_obligor_name_id': 'None',
    'primary_name_abbreviated': rng.choice(
        ['LOS ANG CY CA MET TRA AUT', 'NYC WTR', 'TX TRANSP COMM', 'CHI O HARE', 'CA ST GO'], N),
}, index=rand_cusips(N))
df.index.name = 'CUSIP'

out = HERE / 'Synthetic_Evals.csv'
df.to_csv(out)
print(f"synthetic evals: {len(df):,} rows -> {out}\n")

# ---- drive the real pipeline on it ----
import subprocess, sys
r = subprocess.run([sys.executable, str(HERE / 'run_pipeline.py'),
                    '--wire', str(HERE / 'BAML Write-Up.txt'),
                    '--evals', str(out),
                    '--model', str(HERE / 'model_synth.joblib'),
                    '--issuer', 'LOS ANG',
                    '--retrain', '--report', '--no-archive'],
                   cwd=HERE)
sys.exit(r.returncode)
