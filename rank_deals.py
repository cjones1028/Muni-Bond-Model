"""Risk-adjusted ranking of wire tranches.

For each tranche the model gives a gap (wire yield - model fair yield,
concession included; positive = screens cheap). The gap is then judged
against the RISK of believing it, built from three measured components:

  sigma_bucket : the model's own holdout error (MAE, scaled to stdev) for
                 bonds of this coupon x tenor bucket -- "how wrong is the
                 model usually on bonds LIKE THIS" (computed empirically,
                 cached in bucket_errors.csv, regenerate after retraining)
  sigma_ens    : ensemble disagreement on this specific tranche
  sigma_conc   : uncertainty of the +13bp concession estimate (2 deals -> 3bp)

  net edge     = gap - exit cost (default 5 bps: primary buys at the offering,
                 the cost is the eventual exit half-spread)
  edge ratio   = net / sigma_total       (an information-ratio per tranche)
  P(real)      = Phi(net / sigma_total)  (chance the edge survives the noise)

Ranked by edge ratio. Ratios below ~0.5 are noise dressed as opportunity.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from Wire_Parser import parse_wire

EXIT_COST_BPS = 5.0
CONCESSION_BPS = 14.0   # from concession_tracker.py (implied 13.6 / 14.2 on 2 deals)
SIGMA_CONC = 3.0        # kept conservative while n=2, despite the tight spread
MAE_TO_SD = 1.25        # normal-dist MAE -> stdev
CPN_BINS = [0, 4, 5, 6, 99]
TEN_BINS = [0, 5, 15, 25, 200]


def bucket_error_table():
    """Holdout MAE by coupon x tenor bucket; cached to bucket_errors.csv."""
    cache = HERE / 'bucket_errors.csv'
    if cache.exists():
        return pd.read_csv(cache, index_col=[0, 1])
    print("computing bucket error table from holdout (one-time)...")
    df = mm.load_evals(HERE / 'ICE_Evals.csv')
    df = mm.clean_universe(df)
    dfF = mm.build_features(df)
    bundle = mm.load_bundle(HERE / 'model.joblib')
    rng = np.random.RandomState(42)
    test = rng.rand(len(dfF)) < 0.2
    h = dfF[test].copy()
    X = mm._prep_matrix(h, bundle['numeric'], bundle['categorical'], bundle['categories'])
    h['abs_err'] = np.abs(mm.predict_bundle(bundle, X) - h['target_yield']) * 100
    h['cb'] = pd.cut(pd.to_numeric(h['current_coupon_rate'], errors='coerce'), CPN_BINS)
    h['tb'] = pd.cut(pd.to_numeric(h['days_to_maturity_30_360'], errors='coerce') / 360, TEN_BINS)
    tab = h.groupby(['cb', 'tb'], observed=False)['abs_err'].agg(['mean', 'count'])
    overall = h['abs_err'].mean()
    tab.loc[tab['count'] < 100, 'mean'] = np.nan     # thin buckets -> fallback
    tab['mean'] = tab['mean'].fillna(overall)
    tab.index = pd.MultiIndex.from_tuples([(str(a), str(b)) for a, b in tab.index])
    tab.to_csv(cache)
    return tab


def bond_price(y_pct, cpn_pct, years, redemption=100.0):
    y, c = y_pct / 200, cpn_pct / 2
    n = max(int(round(years * 2)), 1)
    disc = (1 + y) ** -np.arange(1, n + 1)
    return c * disc.sum() + redemption * disc[-1]


def dv01(y_pct, cpn_pct, years):
    return (bond_price(y_pct - 0.005, cpn_pct, years)
            - bond_price(y_pct + 0.005, cpn_pct, years)) / 0.01


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


btab = bucket_error_table()
tdf = pd.read_parquet(HERE / 'template_cache.parquet')
bundle = mm.load_bundle(HERE / 'model.joblib')

DEALS = [('LA Metro I-105 (BBB- toll)', 'BAML Write-Up.txt', 'LOS ANG'),
         ('NYC TFA (AAA-class)', 'NYC_TFA_wire.txt', 'CITY TRANSITIONAL FIN')]

rows = []
for deal_name, fname, issuer in DEALS:
    deal = parse_wire(open(HERE / fname, encoding='utf-8').read())
    templ = mm.template_from(tdf, issuer_contains=issuer)
    res = mm.price_wire(deal, bundle, templ, concession_bps=CONCESSION_BPS)
    settlement = deal['dated_date'] or pd.Timestamp.now().strftime('%m/%d/%Y')
    for mat, r in res.iterrows():
        workout = r['Priced To'] if r['Priced To'] != 'Maturity' else mat
        yrs_wo = mm.days_30_360(settlement, workout) / 360
        yrs_mat = mm.days_30_360(settlement, mat) / 360
        d = dv01(r['Wire Yield'], r['Coupon'], yrs_wo)

        gap = -r['Error (bps)']                      # + = screens cheap
        cb = pd.cut([r['Coupon']], CPN_BINS)[0]
        tb = pd.cut([yrs_mat], TEN_BINS)[0]
        s_bucket = float(btab.loc[(str(cb), str(tb)), 'mean']) * MAE_TO_SD
        s_ens = float(r['Model Std (bps)'])
        s_tot = math.sqrt(s_bucket**2 + s_ens**2 + SIGMA_CONC**2)

        net = gap - EXIT_COST_BPS
        ratio = net / s_tot
        # THE number: expected profit minus a risk charge of half the total
        # uncertainty, in $ per $1mm face. Positive = edge outruns the risk.
        profit_score = int(round((net - 0.5 * s_tot) / 100 * d * 10000))
        rows.append({
            'PROFIT ($/mm)': profit_score,
            'Deal': deal_name.split(' (')[0], 'Maturity': mat, 'Coupon': r['Coupon'],
            'Cheap (bps)': round(gap, 1), 'Net edge': round(net, 1),
            'σ model': round(s_bucket, 1), 'σ ens': round(s_ens, 1),
            'σ total': round(s_tot, 1), 'Edge ratio': round(ratio, 2),
            'P(real)': f"{phi(ratio)*100:.0f}%",
            'EV $/mm': int(round(net / 100 * d * 10000)),
            '1σ loss $/mm': int(round((s_tot - net) / 100 * d * 10000)),
            'Size ($mm)': round(r['Amount ($)'] / 1e6, 1),
        })

out = pd.DataFrame(rows).sort_values('PROFIT ($/mm)', ascending=False)
out.to_csv(HERE / 'deal_ranking.csv', index=False)

simple = out[['Deal', 'Maturity', 'Coupon', 'PROFIT ($/mm)', 'Size ($mm)']]
print(simple.to_string(index=False))
print("\nPROFIT = risk-adjusted expected profit per $1mm face "
      "(edge minus exit cost minus a half-sigma risk charge, x DV01).")
print("Positive = the edge outruns the risk. Negative = fairly priced; "
      "the deal's value is the concession itself, nothing extra.")
print(f"full detail (uncertainty components, P(real), EV, downside) -> deal_ranking.csv")
print(f"assumptions: exit cost {EXIT_COST_BPS}bps | concession {CONCESSION_BPS:.0f}±{SIGMA_CONC}bps | "
      f"risk charge 0.5 sigma | bucket errors from holdout")
