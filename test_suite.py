"""
test_suite -- verification of the wire-pricing pipeline.

    python test_suite.py

Checks, in order:
  1. 30/360 day-count math against hand-computed cases
  2. BAML wire parse: every deal-level field and spot-checked tranches
     against values read directly off the wire text
  3. Loop Capital (NYC TFA) wire parse: same treatment, other dialect
  4. Edge cases: no call features, maturity == call date, missing price line
  5. Rating canonicalization: wire-supplied ratings must reach the model
     (not silently become missing)
  6. Model determinism: same input twice -> identical predictions
  7. End-to-end regression: both wires price with sane error vs dealer

Exits non-zero on any failure. Run after ANY code change.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from Wire_Parser import parse_wire  # noqa: E402
import muni_model as mm  # noqa: E402

FAILURES = []


def check(name, cond, detail=''):
    status = 'ok  ' if cond else 'FAIL'
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not cond else ''))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


# ------------------------------------------------------------ 1. day count
print("1. 30/360 day count")
check("9/3/26 -> 6/1/36 = 3508", mm.days_30_360('09/03/2026', '06/01/2036') == 3508)
check("1/31 -> 7/31 = 180 (both EOM)", mm.days_30_360('01/31/2026', '07/31/2026') == 180)
check("1/30 -> 2/28 = 28", mm.days_30_360('01/30/2026', '02/28/2026') == 28)
check("same day = 0", mm.days_30_360('06/01/2036', '06/01/2036') == 0)
check("9/3/26 -> 6/1/45 = 6748", mm.days_30_360('09/03/2026', '06/01/2045') == 6748)

# ------------------------------------------------------------ 2. BAML wire
print("2. BAML wire parse")
baml = parse_wire(open(HERE / 'BAML Write-Up.txt', encoding='utf-8').read())
check("deal size 509,145,000", baml['issue_amount'] == 509_145_000)
check("16 tranches", len(baml['tranches']) == 16)
check("tranche sum == deal size", sum(t['amount'] for t in baml['tranches']) == 509_145_000)
check("Moody's Baa2 -> 9", baml['moody'] == 'Baa2' and baml['moody_rating'] == 9)
check("S&P NR falls back to Fitch BBB- -> 10", baml['sandp'] == 'NR' and baml['sandp_rating'] == 10)
check("dated 09/03/2026", baml['dated_date'] == '09/03/2026')
check("call 06/01/2036 @ 100", baml['call_date'] == '06/01/2036' and baml['call_price'] == 100.0)

t34 = baml['tranches'][0]
check("2034: 5,650M / 5% / 3.57 / 109.595 / no PTC",
      t34['maturity'] == '06/01/2034' and t34['amount'] == 5_650_000
      and t34['coupon'] == 5.0 and t34['yield'] == 3.57
      and t34['price'] == 109.595 and t34['ptc_date'] is None)
t44 = next(t for t in baml['tranches'] if t['maturity'] == '06/01/2044')
check("2044: 17,965M / 4.54 / 103.583 / PTC 06/01/2036",
      t44['amount'] == 17_965_000 and t44['yield'] == 4.54
      and t44['price'] == 103.583 and t44['ptc_date'] == '06/01/2036')
t56s = [t for t in baml['tranches'] if t['maturity'] == '06/01/2056']
check("two 2056 tranches (5.0 disc / 5.25 PTC)",
      len(t56s) == 2
      and {t['coupon'] for t in t56s} == {5.0, 5.25}
      and next(t for t in t56s if t['coupon'] == 5.0)['price'] == 98.319
      and next(t for t in t56s if t['coupon'] == 5.0)['ptc_date'] is None
      and next(t for t in t56s if t['coupon'] == 5.25)['price'] == 101.439)
check("takedown parsed 0.40", all(t['takedown'] == 0.40 for t in baml['tranches']))
check("description has issuer", 'LOS ANGELES' in baml['description'])

# ------------------------------------------------------------ 3. NYC TFA wire
print("3. Loop Capital (NYC TFA) wire parse")
nyc = parse_wire(open(HERE / 'NYC_TFA_wire.txt', encoding='utf-8').read())
check("deal size 1,500,000,000 from Subject", nyc['issue_amount'] == 1_500_000_000)
check("7 tranches despite orders/spread tokens", len(nyc['tranches']) == 7)
check("tranche par sum 926,405,000",
      sum(t['amount'] for t in nyc['tranches']) == 926_405_000)
t46 = nyc['tranches'][0]
check("2046: 84,630M / 5.25 / 4.57 / 105.478 / PTC 11/01/2036",
      t46['amount'] == 84_630_000 and t46['coupon'] == 5.25
      and t46['yield'] == 4.57 and t46['price'] == 105.478
      and t46['ptc_date'] == '11/01/2036')
t53 = next(t for t in nyc['tranches'] if t['coupon'] == 5.0)
check("2053 5.00%: 122,075M / 4.93 / 100.547",
      t53['maturity'] == '11/01/2053' and t53['amount'] == 122_075_000
      and t53['yield'] == 4.93 and t53['price'] == 100.547)
check("no ratings on wire -> None (template will supply)",
      nyc['moody_rating'] is None and nyc['sandp_rating'] is None)
check("call 11/01/2036 @ 100", nyc['call_date'] == '11/01/2036' and nyc['call_price'] == 100.0)
check("no DATED line -> None", nyc['dated_date'] is None)
check("description from Subject", 'Transitional' in nyc['description'])

# ------------------------------------------------------------ 4. edge cases
print("4. edge cases")
no_call = parse_wire("""RE: $ 10,000,000
TEST ISSUER
MOODY'S: Aa2 (Stable)
DATED:01/15/2027
06/01/2030     10,000M     5.00%     3.00      0.25
                      (Approx. $ Price 106.100)
""")
check("no CALL FEATURES -> call_date None", no_call['call_date'] is None)
check("single tranche parsed", len(no_call['tranches']) == 1
      and no_call['tranches'][0]['price'] == 106.100)
check("Aa2 -> 3", no_call['moody_rating'] == 3)

at_call = parse_wire("""RE: $ 5,000,000
TEST
DATED:01/15/2027
06/01/2036      5,000M     5.00%     3.50      0.25
                      (Approx. $ Price 104.000)
CALL FEATURES:  Optional call in 06/01/2036 @ 100.00
""")
from datetime import datetime as _dt
mat_eq_call = (_dt.strptime(at_call['tranches'][0]['maturity'], '%m/%d/%Y')
               > _dt.strptime(at_call['call_date'], '%m/%d/%Y'))
check("maturity == call date -> NOT callable", mat_eq_call is False)

no_price = parse_wire("""RE: $ 5,000,000
TEST
DATED:01/15/2027
06/01/2030      5,000M     5.00%     3.00      0.25
""")
check("missing price line -> price None (skipped later, not crash)",
      no_price['tranches'][0]['price'] is None)

# ------------------------------------------------------------ 5. rating canonicalization
print("5. categorical canonicalization")
c = mm._canon_cat(pd.Series([9, '9', 9.0, 'True', np.nan, 'AA+']))
check("9 / '9' / 9.0 all -> '9.0'", list(c[:3]) == ['9.0', '9.0', '9.0'])
check("'True' stays 'True'", c.iloc[3] == 'True')
check("'AA+' stays 'AA+'", c.iloc[5] == 'AA+')

# ------------------------------------------------------------ 6/7. model + end-to-end
print("6. model training, determinism, rating reach")
import os
df = mm.load_evals(HERE / 'ICE_Evals.csv')
df = mm.clean_universe(df)
if os.environ.get('TEST_REUSE_MODEL') == '1':
    print("  (TEST_REUSE_MODEL=1: validating the existing model.joblib, no retrain)")
    bundle = mm.load_bundle(HERE / 'model.joblib')
else:
    bundle = mm.train_yield_model(df, seed=42)
    mm.save_bundle(bundle, HERE / 'model.joblib')
check("holdout MAE < 10 bps", bundle['mae_bps'] < 10, f"{bundle['mae_bps']:.1f}")

dfF = mm.build_features(df)
Xa = mm._prep_matrix(dfF.head(500), bundle['numeric'], bundle['categorical'], bundle['categories'])
p1, p2 = mm.predict_bundle(bundle, Xa), mm.predict_bundle(bundle, Xa)
check("deterministic predictions", bool(np.array_equal(p1, p2)))

reloaded = mm.load_bundle(HERE / 'model.joblib')
p3 = mm.predict_bundle(reloaded, mm._prep_matrix(dfF.head(500), reloaded['numeric'],
                                                 reloaded['categorical'], reloaded['categories']))
check("save/reload identical predictions", bool(np.allclose(p1, p3)))

# rating override must land in a real category, not NaN
row = pd.DataFrame([{'normalized_moody_long_rating': 9}])
Xr = mm._prep_matrix(row, [], ['normalized_moody_long_rating'], bundle['categories'])
check("wire rating 9 reaches model as '9.0' category",
      not pd.isna(Xr['normalized_moody_long_rating'].iloc[0]),
      "override became missing -- canonicalization broken")

print("7. end-to-end regression on both wires")
# broad template (production usage): the I-105 toll deal is a separate credit
# from MTA's sales-tax bonds, so a narrow same-issuer template anchors to the
# wrong credit (verified: ~33 bps one-directional skew). price_wire warns on
# that mismatch; the broad pool is correct here.
templ_la = mm.template_from(df, issuer_contains='LOS ANG')
res_la = mm.price_wire(baml, bundle, templ_la, concession_bps=13)
la_mae = res_la['Error (bps)'].abs().mean()
check("LA Metro: 16 rows priced", len(res_la) == 16)
check("LA Metro mean abs err < 8 bps (was ~3)", la_mae < 8, f"{la_mae:.1f}")

templ_bad = mm.template_from(df, issuer_contains='LOS ANG CY CA MET TRA AUT')
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    mm.price_wire(baml, bundle, templ_bad, concession_bps=13)
check("credit-mismatch template triggers warning", 'WARNING: wire rating' in buf.getvalue())

templ_ny = mm.template_from(df, issuer_contains='CITY TRANSITIONAL FIN')
res_ny = mm.price_wire(nyc, bundle, templ_ny, concession_bps=13)
ny_mae = res_ny['Error (bps)'].abs().mean()
check("NYC TFA: 7 rows priced", len(res_ny) == 7)
check("NYC TFA mean abs err < 8 bps (was ~3)", ny_mae < 8, f"{ny_mae:.1f}")
check("all predictions finite", bool(np.isfinite(res_la['Model Yield']).all()
                                     and np.isfinite(res_ny['Model Yield']).all()))

# ------------------------------------------------------------ verdict
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
