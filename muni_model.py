"""
muni_model -- reconstructed muni new-issue pricing model.

Standalone reimplementation of the Curve_Analysis pipeline (which lives on the
b.fund machine), upgraded per the accuracy plan:

  * trains on the WHOLE filtered universe from ICE_Evals.csv, not one issuer
    (issuer becomes a feature via categoricals, not a filter)
  * gradient boosting (LightGBM) instead of a linear model
  * holdout evaluation with error reported in bps, bucketed by tenor /
    callability / coupon so you can see WHERE the model misses
  * a new-issue concession parameter to net out the secondary-eval vs
    primary-wire gap

Intended flow (see run_pipeline.py for the one-command version):

    df     = load_evals('ICE_Evals.csv')
    df     = clean_universe(df, state='California')
    bundle = train_yield_model(df)
    error_report(bundle, df)

    deal     = parse_wire(open('BAML Write-Up.txt').read())   # Wire_Parser
    template = template_from(df, issuer_contains='LOS ANG')
    results  = price_wire(deal, bundle, template)
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

from lightgbm import LGBMRegressor


# ---------------------------------------------------------------- day count

def days_30_360(start, end):
    """30/360 US day count between two dates (str 'm/d/Y' or datetime)."""
    if isinstance(start, str):
        start = datetime.strptime(start, '%m/%d/%Y')
    if isinstance(end, str):
        end = datetime.strptime(end, '%m/%d/%Y')
    d1, d2 = min(start.day, 30), end.day
    if d2 == 31 and d1 == 30:
        d2 = 30
    return (360 * (end.year - start.year)
            + 30 * (end.month - start.month)
            + (d2 - d1))


# ---------------------------------------------------------------- features

# numeric features, matching Run Code Part 1 where the eval file provides them
NUMERIC_FEATURES = [
    'current_coupon_rate', 'days_to_first_call_30_360', 'days_to_maturity_30_360',
    'days_to_next_call_30_360', 'days_to_refund_30_360',
    'days_from_call_to_maturity_30_360', 'issue_amount', 'next_call_price',
    'days_to_next_sink_date_30_360', 'next_sink_price', 'outstanding_amount',
    'principal_factor', 'log_issue_amount', 'log_outstanding_amount',
    'outstanding_pct_of_issue', 'call_moneyness', 'call_moneyness_pct',
    'above_call_price', 'par_dv01',
]
# par_dv01: analytic duration of a par bond with this coupon/maturity --
# computed from STRUCTURE only (identical for a wire tranche and a seasoned
# bond, no quotes involved), unlike ice_dv01 which derives from the bond's
# own bid/ask and is used only as a training FILTER (min_dv01).
# Tested and REJECTED as features (exp_config.py, 2026-08-19): issue_price,
# min_denom_amount, denom_increment_amount, bond_age_days, days_to_next_coupon.
# They left the universe holdout unchanged but degraded new-issue wire pricing
# ~3x (NYC TFA 3.3 -> 9.1 bps): their values for a brand-new bond (age 0,
# missing denominations, issue_price == today's price) sit outside the
# training distribution, so the model extrapolates badly exactly where we
# need it. Do not re-add without re-running exp_config.py on real wires.

CATEGORICAL_FEATURES = [
    'ad_valorem_tax_status_desc', 'bond_insurance', 'composite_rating',
    'coupon_type_desc', 'distinct_call_timing_desc', 'enhanced_composite_rating',
    'federal_tax_status_desc', 'incorporated_state_code_desc',
    'muni_security_type_desc', 'normalized_moody_enhanced_long_rating',
    'normalized_moody_long_rating', 'normalized_sandp_long_rating',
    'purpose_class_desc', 'purpose_sub_class_desc', 'security_class',
    'state_tax_status_desc', 'use_of_proceeds_desc', 'organization_master_id',
    'conduit_obligor_name_id', 'bank_qualified', 'call_notice', 'call_indicator',
    'called_redemption_type_desc',   # pre-refunded / escrowed / called status:
                                     # these trade off the escrow, not the credit
]


def load_evals(path):
    """Read an ICE evals CSV (as written by Pull_ICE_Evals.ps1 or the
    notebooks), indexed by CUSIP, with a 'target_yield' column added."""
    df = pd.read_csv(path, index_col=0, low_memory=False)

    # target: the feed's own mid yield when present, else bid/offer average
    yb = pd.to_numeric(df.get('yield_bid'), errors='coerce')
    yo = pd.to_numeric(df.get('yield_offer'), errors='coerce')
    ym = pd.to_numeric(df.get('yield_mid'), errors='coerce')
    df['target_yield'] = ym
    df.loc[df['target_yield'].isna(), 'target_yield'] = (yb + yo) / 2
    df.loc[df['target_yield'].isna(), 'target_yield'] = yb

    # eval-side DV01 proxy, same construction as the notebooks
    bid = pd.to_numeric(df.get('bid'), errors='coerce')
    offer = pd.to_numeric(df.get('offer'), errors='coerce')
    with np.errstate(divide='ignore', invalid='ignore'):
        df['ice_dv01'] = (offer - bid) / (yb - yo)

    return df


def clean_universe(df, federal_tax='Tax-exempt', state=None,
                   fixed_rate_only=True, drop_defaulted=True, min_dv01=None):
    """Filter to the modelable universe. Prints what each filter removes so
    nothing disappears silently."""
    n0 = len(df)
    mask = df['target_yield'].notna() & (df['target_yield'] > 0)
    print(f"no/zero yield:      -{(~mask).sum():,}")

    if federal_tax and 'federal_tax_status_desc' in df:
        m = df['federal_tax_status_desc'].eq(federal_tax)
        print(f"not {federal_tax}:   -{(mask & ~m).sum():,}")
        mask &= m
    if fixed_rate_only and 'coupon_type_desc' in df:
        m = df['coupon_type_desc'].eq('Fixed rate')
        print(f"not fixed rate:     -{(mask & ~m).sum():,}")
        mask &= m
    if drop_defaulted and 'default_indicator' in df:
        m = ~df['default_indicator'].astype(str).str.lower().isin(['true', '1', '1.0'])
        print(f"defaulted:          -{(mask & ~m).sum():,}")
        mask &= m
    if 'instrument_status_desc' in df:
        # Dead = matured / fully called / defunct -- their marks are stale and
        # poison both training and screening
        m = ~df['instrument_status_desc'].eq('Dead')
        print(f"dead instruments:   -{(mask & ~m).sum():,}")
        mask &= m
    if min_dv01 is not None and 'ice_dv01' in df:
        # quote-derived DV01 as a data-quality FILTER (the original Part 1
        # used min_dv01=1): drops ultra-short / oddly-quoted bonds whose
        # marks are noisiest. NaN dv01 (no two-sided quote) also drops.
        d = pd.to_numeric(df['ice_dv01'], errors='coerce')
        m = d >= min_dv01
        print(f"dv01 < {min_dv01} or unquoted: -{(mask & ~m.fillna(False)).sum():,}")
        mask &= m.fillna(False)
    if state and 'incorporated_state_code_desc' in df:
        m = df['incorporated_state_code_desc'].eq(state)
        print(f"not {state}:  -{(mask & ~m).sum():,}")
        mask &= m

    out = df[mask].copy()
    print(f"universe: {n0:,} -> {len(out):,}")
    return out


def stacked_frame(here=None):
    """THE canonical training frame: every dated snapshot in evals_archive
    stacked (fallback: the single current evals file). All trainers must use
    this -- three separate code paths once trained different recipes and two
    of them silently overwrote the production model (caught 8/27)."""
    here = Path(here) if here else Path(__file__).resolve().parent
    snaps = sorted((here / 'evals_archive').glob('ICE_Evals_*.csv'))
    if snaps:
        df = pd.concat([clean_universe(load_evals(p)) for p in snaps])
        print(f"stacked {len(snaps)} snapshot(s): {len(df):,} rows")
        return df
    return clean_universe(load_evals(here / 'ICE_Evals.csv'))


def build_features(df):
    """Derived columns (mirrors Run Code Part 1's derived features)."""
    df = df.copy()
    issue = pd.to_numeric(df.get('issue_amount'), errors='coerce')
    outst = pd.to_numeric(df.get('outstanding_amount'), errors='coerce')
    df['log_issue_amount'] = np.log1p(issue)
    df['log_outstanding_amount'] = np.log1p(outst)
    df['outstanding_pct_of_issue'] = outst / issue

    # analytic par-bond duration (points per 1% yield), pure structure:
    # ModDur_par = [1 - (1+c/2)^(-2T)] / c   with c = coupon, T = years
    cpn = pd.to_numeric(df.get('current_coupon_rate'), errors='coerce') / 100
    T = pd.to_numeric(df.get('days_to_maturity_30_360'), errors='coerce') / 360
    with np.errstate(all='ignore'):
        df['par_dv01'] = np.where(cpn > 0, (1 - (1 + cpn / 2) ** (-2 * T)) / cpn, T)

    mid = pd.to_numeric(df.get('mid'), errors='coerce')
    ncp = pd.to_numeric(df.get('next_call_price'), errors='coerce')
    df['call_moneyness'] = (mid - ncp).fillna(0.0)
    df['call_moneyness_pct'] = (df['call_moneyness'] / ncp).fillna(0.0)
    df['above_call_price'] = (mid > ncp).astype(float)

    # age and coupon-cycle position (skipped gracefully when cols are absent,
    # e.g. for wire tranches, where price_wire sets them directly)
    def _dt_col(name):
        return pd.to_datetime(df[name], errors='coerce') if name in df else None
    snap, issue_dt, nxt_cpn = _dt_col('model_date'), _dt_col('issue_date'), _dt_col('next_coupon_payment_date')

    # the feed's days_to_refund_30_360 arrives empty, but refund_date is
    # populated for ~8k pre-refunded/escrowed/called bonds -- rebuild the
    # feature so the model can price them off the escrow date
    refund_dt = _dt_col('refund_date')
    if refund_dt is not None and snap is not None:
        d2r = (pd.to_numeric(df['days_to_refund_30_360'], errors='coerce')
               if 'days_to_refund_30_360' in df else pd.Series(np.nan, index=df.index))
        if d2r.isna().all():
            df['days_to_refund_30_360'] = (refund_dt - snap).dt.days
    if 'bond_age_days' not in df:
        df['bond_age_days'] = ((snap - issue_dt).dt.days
                               if snap is not None and issue_dt is not None else np.nan)
    if 'days_to_next_coupon' not in df:
        df['days_to_next_coupon'] = ((nxt_cpn - snap).dt.days
                                     if snap is not None and nxt_cpn is not None else np.nan)
    return df


def _canon_cat(s):
    """Canonical string form for a categorical column, so training and
    prediction always agree: any numeric-looking value is rendered as a float
    string ('9' and 9 and 9.0 all become '9.0'); everything else is str().
    Without this, a wire-supplied rating of 9 (str '9') would not match the
    training category '9.0' and would silently turn into a missing value."""
    out = pd.Series(s).astype(str)
    as_num = pd.to_numeric(out, errors='coerce')
    mask = as_num.notna()
    return out.where(~mask, as_num.astype(float).astype(str))


def _prep_matrix(df, numeric, categorical, categories=None):
    """Assemble the model matrix; categoricals as pandas category dtype so
    LightGBM handles them natively. `categories` (from training) pins the
    category sets at prediction time."""
    X = pd.DataFrame(index=df.index)
    for c in numeric:
        X[c] = pd.to_numeric(df[c], errors='coerce') if c in df else np.nan
    for c in categorical:
        if c in df:
            s = _canon_cat(df[c])
            s.index = df.index
        else:
            s = pd.Series('missing', index=df.index)
        if categories is not None:
            X[c] = pd.Categorical(s, categories=categories[c])
        else:
            X[c] = s.astype('category')
    return X


# ---------------------------------------------------------------- training

def predict_bundle(bundle, X):
    """Average prediction across the bundle's ensemble members."""
    models = bundle.get('models') or [bundle['model']]
    return np.mean([m.predict(X) for m in models], axis=0)


def train_yield_model(df, numeric=None, categorical=None,
                      test_size=0.2, seed=42, n_ensemble=3, **lgbm_kwargs):
    """Train the yield model on cleaned+featured evals. Returns a bundle dict
    with the ensemble, feature lists, pinned categories, holdout frame, and
    holdout MAE in bps.

    Uses the L1 (absolute error) objective -- the metric we actually care
    about -- with early stopping on a validation slice carved out of the
    training portion (the test holdout stays untouched), and averages
    n_ensemble models trained with different seeds.

    NOTE: the split is random because one eval pull is a single cross-section.
    Once daily pulls accumulate, switch to time-based splits (train on past
    days, test on the latest) -- random splits on panel data flatter the model.
    """
    import lightgbm as lgb

    # sub-1-year bonds are excluded from TRAINING (adopted 8/26 after
    # exp_shortdrop/exp_dv01/exp_combo): their erratic marks polluted split
    # decisions for the whole curve -- dropping them cut the >=1yr holdout
    # median from 2.24 to 0.81 bps with no net wire degradation. Consequence:
    # the model has no support under 1yr, so price_wire marks sub-1yr
    # tranches as reduced confidence.
    if 'days_to_maturity_30_360' in df:
        tenor = pd.to_numeric(df['days_to_maturity_30_360'], errors='coerce')
        n0 = len(df)
        df = df[tenor >= 360]
        print(f"training filter: dropped {n0 - len(df):,} sub-1yr bonds -> {len(df):,}")

    df = build_features(df)

    numeric = [c for c in (numeric or NUMERIC_FEATURES) if c in df.columns]
    categorical = [c for c in (categorical or CATEGORICAL_FEATURES) if c in df.columns]
    missing = sorted(set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
                     - set(numeric) - set(categorical))
    if missing:
        print(f"features not in eval file (skipped): {missing}")

    X = _prep_matrix(df, numeric, categorical)
    y = df['target_yield'].astype(float)

    # CUSIP-level split (8/27 fix): stacked snapshots repeat CUSIPs, so the
    # old row-level split let ~80% of "held-out" bonds also reach training
    # via their other snapshot's row. Split by CUSIP, and persist the set in
    # the bundle so downstream evaluations grade on genuinely unseen bonds.
    cusips = pd.Index(df.index.unique())
    rc = np.random.RandomState(seed).rand(len(cusips))
    test_cusips = set(cusips[rc < test_size])
    val_cusips = set(cusips[(rc >= test_size) & (rc < test_size + 0.08)])
    test_mask = df.index.isin(test_cusips)
    val_mask = df.index.isin(val_cusips)
    fit_mask = ~test_mask & ~val_mask

    params = dict(objective='l1', n_estimators=8000, learning_rate=0.03,
                  num_leaves=127, min_child_samples=30, subsample=0.9,
                  subsample_freq=1, colsample_bytree=0.9, verbose=-1)
    params.update(lgbm_kwargs)

    models = []
    for k in range(n_ensemble):
        m = LGBMRegressor(**{**params, 'random_state': seed + k})
        m.fit(X[fit_mask], y[fit_mask],
              eval_set=[(X[val_mask], y[val_mask])], eval_metric='l1',
              callbacks=[lgb.early_stopping(150, verbose=False)])
        models.append(m)

    bundle = {'models': models, 'model': models[0],
              'numeric': numeric, 'categorical': categorical,
              'holdout_cusips': sorted(test_cusips),
              'categories': {c: list(X[c].cat.categories) for c in categorical}}

    pred = predict_bundle(bundle, X[test_mask])
    err_bps = (pred - y[test_mask].to_numpy()) * 100
    mae = float(np.mean(np.abs(err_bps)))
    med = float(np.median(np.abs(err_bps)))
    print(f"holdout ({test_mask.sum():,} bonds): MAE {mae:.1f} bps | median {med:.1f} bps "
          f"| ensemble of {n_ensemble}, best_iter {[m.best_iteration_ for m in models]}")

    holdout = df[test_mask].copy()
    holdout['pred_yield'] = pred
    holdout['err_bps'] = err_bps
    bundle.update({'holdout': holdout, 'mae_bps': mae, 'median_bps': med})
    return bundle


def error_report(bundle, top_n=12):
    """Where does the model miss? Buckets holdout error by tenor, callability,
    and coupon. This is what tells you what to fix next."""
    h = bundle['holdout']
    print("\n--- holdout error by bucket (mean abs bps / count) ---")

    tenor_yrs = pd.to_numeric(h.get('days_to_maturity_30_360'), errors='coerce') / 360
    buckets = pd.cut(tenor_yrs, [0, 2, 5, 10, 15, 20, 30, 100])
    print("\nby tenor (yrs):")
    print(h.groupby(buckets, observed=True)['err_bps']
          .agg(mae=lambda s: s.abs().mean(), n='count').round(1))

    if 'call_indicator' in h:
        print("\nby callability:")
        print(h.groupby(h['call_indicator'].astype(str))['err_bps']
              .agg(mae=lambda s: s.abs().mean(), n='count').round(1))

    cpn = pd.to_numeric(h.get('current_coupon_rate'), errors='coerce')
    print("\nby coupon:")
    print(h.groupby(pd.cut(cpn, [0, 3, 4, 5, 6, 20]), observed=True)['err_bps']
          .agg(mae=lambda s: s.abs().mean(), n='count').round(1))

    print(f"\nworst {top_n} holdout misses:")
    cols = [c for c in ['target_yield', 'pred_yield', 'err_bps',
                        'current_coupon_rate', 'days_to_maturity_30_360'] if c in h]
    print(h.reindex(h['err_bps'].abs().sort_values(ascending=False).index)[cols].head(top_n).round(2))


def confidence_flags(df, pred_yield, ens_std_bps):
    """Trust filter for using model-vs-market gaps as SIGNALS (secondary
    screening). Every flag is computable before knowing the true yield.

    Validated on the seed-42 holdout (diag_confidence.py, 2026-08-19):
    trusted 65% of bonds -> mean err 3.5 / median 1.7 bps; the filter caught
    89% of all >50bp errors and 95% of >100bp errors in advance. A large
    model-vs-market gap on a FLAGGED bond usually means the market knows
    something the model can't see (distress, escrow, illiquidity) -- treat it
    as a research lead, never a trade signal.

    Returns (flags DataFrame, trusted boolean Series).
    """
    idx = df.index
    flags = pd.DataFrame(index=idx)
    rating = (pd.to_numeric(df['composite_rating'], errors='coerce')
              if 'composite_rating' in df else pd.Series(np.nan, index=idx))
    flags['unrated'] = rating.isna()
    flags['junk_rated'] = rating > 10
    flags['pred_yield_high'] = pd.Series(np.asarray(pred_yield), index=idx) > 5.5
    flags['conduit'] = (df['conduit_obligor_name_id'].notna()
                        if 'conduit_obligor_name_id' in df else False)
    flags['micro'] = (pd.to_numeric(df['outstanding_amount'], errors='coerce') < 2_000_000
                      if 'outstanding_amount' in df else False)
    flags['ens_disagree'] = pd.Series(np.asarray(ens_std_bps), index=idx) > 5
    flags['ultra_short'] = (pd.to_numeric(df['days_to_maturity_30_360'], errors='coerce') < 540
                            if 'days_to_maturity_30_360' in df else False)
    return flags, ~flags.any(axis=1)


def save_bundle(bundle, path):
    slim = {k: v for k, v in bundle.items() if k != 'holdout'}
    joblib.dump(slim, path)
    print(f"model saved -> {path}")


def load_bundle(path):
    return joblib.load(path)


# ---------------------------------------------------------------- wire pricing

def template_from(df, issuer_contains=None, cusip=None):
    """Categorical defaults for a new issue, taken from comparable existing
    bonds: a specific CUSIP, or the modal values across all bonds whose
    issuer/obligor text matches `issuer_contains`."""
    if cusip is not None:
        row = df.loc[cusip]
        if isinstance(row, pd.DataFrame):   # duplicated index entry
            row = row.iloc[0]
        return {c: str(row[c]) for c in CATEGORICAL_FEATURES if c in df.columns}

    pool = df
    if issuer_contains:
        text_cols = [c for c in ['primary_name_abbreviated', 'organization_master_id',
                                 'conduit_obligor_name_id'] if c in df.columns]
        m = pd.Series(False, index=df.index)
        for c in text_cols:
            m |= df[c].astype(str).str.contains(issuer_contains, case=False, na=False)
        if m.any():
            pool = df[m]
            print(f"template: {m.sum():,} bonds match issuer '{issuer_contains}'")
        else:
            print(f"template: no issuer match for '{issuer_contains}', using full-universe modes")
    return {c: str(pool[c].mode(dropna=False).iloc[0])
            for c in CATEGORICAL_FEATURES if c in pool.columns}


def price_wire(deal, bundle, template, settlement_date=None,
               concession_bps=0.0, rating_overrides=True):  # 0.0 = API default: caller supplies calibration
    """Price every tranche of a parsed wire (Wire_Parser.parse_wire output).
    Returns a DataFrame of wire vs model yields with error in bps.

    concession_bps: expected new-issue concession -- added to the model's
    secondary-market yield before comparing to the wire. Estimate it from the
    archive once a few deals accumulate; 0 until then.
    """
    settlement = settlement_date or deal['dated_date']
    if not settlement:
        settlement = datetime.now().strftime('%m/%d/%Y')
        print(f"WARNING: wire has no DATED line and no settlement_date passed -- "
              f"assuming today ({settlement}). Pass --settlement for precision.")
    call_date, call_price = deal['call_date'], deal['call_price']

    # guard: if the wire's rating is far from the template pool's rating, the
    # template is probably anchored to a DIFFERENT credit of the same issuer
    # (e.g. an issuer's AA sales-tax bonds as template for its BBB toll deal).
    # The model leans on issuer identity, so a credit-mismatched template
    # skews every tranche the same direction.
    tmpl_moody = pd.to_numeric(pd.Series([template.get('normalized_moody_long_rating')]),
                               errors='coerce').iloc[0]
    if deal.get('moody_rating') is not None and pd.notna(tmpl_moody):
        gap = abs(deal['moody_rating'] - tmpl_moody)
        if gap >= 3:
            print(f"WARNING: wire rating ({deal['moody_rating']}) is {gap:.0f} notches from "
                  f"the template pool's typical rating ({tmpl_moody:.0f}). The template may be "
                  f"a different credit of this issuer -- consider a broader --issuer match "
                  f"or a --template-cusip from a comparable credit.")

    rows = []
    for t in deal['tranches']:
        if t['price'] is None:
            continue
        dtm = days_30_360(settlement, t['maturity'])
        callable_ = (call_date is not None and
                     datetime.strptime(t['maturity'], '%m/%d/%Y')
                     > datetime.strptime(call_date, '%m/%d/%Y'))
        dtc = days_30_360(settlement, call_date) if callable_ else np.nan

        feat = dict(template)
        feat.update({
            'current_coupon_rate': t['coupon'],
            'days_to_maturity_30_360': dtm,
            'days_to_first_call_30_360': dtc,
            'days_to_next_call_30_360': dtc,
            'days_from_call_to_maturity_30_360': dtm - dtc if callable_ else np.nan,
            'issue_amount': deal['issue_amount'],
            'outstanding_amount': t['amount'],
            'next_call_price': call_price if callable_ else np.nan,
            'principal_factor': 1.0,
            'mid': t['price'],
            'call_indicator': str(callable_),
            'issue_price': t['price'],
            'bond_age_days': 0.0,
            'days_to_next_coupon': (days_30_360(settlement, deal['first_coupon'])
                                    if deal.get('first_coupon') else np.nan),
        })
        # only override template ratings if the wire actually carried ratings
        if rating_overrides and deal.get('moody_rating') is not None:
            feat['normalized_moody_long_rating'] = str(deal['moody_rating'])
        if rating_overrides and deal.get('sandp_rating') is not None:
            feat['normalized_sandp_long_rating'] = str(deal['sandp_rating'])
        rows.append(feat)

    if not rows:
        raise ValueError(
            "No priced tranches parsed from this wire. Usual causes: the "
            "WIRE_TEXT cell still contains the placeholder text, the paste "
            "was incomplete, or this dealer's format needs a parser "
            "extension. Check the wire text and re-run.")
    fdf = build_features(pd.DataFrame(rows))
    X = _prep_matrix(fdf, bundle['numeric'], bundle['categorical'], bundle['categories'])
    # version guard: if the loaded model expects a derived feature that this
    # code version didn't compute (all-NaN), predictions would silently
    # collapse (seen 8/26: stale kernel + new model -> flat curve, -180bp
    # errors). Refuse loudly instead.
    for _c in ('par_dv01', 'call_moneyness', 'log_issue_amount'):
        if _c in bundle['numeric'] and _c in X and X[_c].isna().all():
            raise RuntimeError(
                f"MODEL/CODE VERSION MISMATCH: the model expects feature '{_c}' "
                f"but this code produced none. You are running STALE code "
                f"against a newer model.joblib -- restart the Jupyter kernel "
                f"(or importlib.reload(muni_model)), or use Price Wire.bat.")
    members = np.array([m.predict(X) for m in (bundle.get('models') or [bundle['model']])])
    # the new-issue concession lives in the LONG bonds -- short tranches get
    # almost none (observed on PSU and Portland: +18-22bp false "rich" reads
    # at 1yr under a flat concession). Ramp: zero at 0yr, full from 8yrs.
    # Keyed on years to WORKOUT (call date if priced-to-call, else maturity)
    # -- the same axis the calibration measures on (unified 8/27).
    workout_days = np.array([
        days_30_360(settlement, t['ptc_date'] or t['maturity'])
        for t in deal['tranches'] if t['price'] is not None], dtype=float)
    ramp = np.clip(workout_days / 360.0 / 8.0, 0, 1)
    pred = members.mean(axis=0) + concession_bps * ramp / 100
    ens_std_bps = members.std(axis=0) * 100

    priced = [t for t in deal['tranches'] if t['price'] is not None]
    out = pd.DataFrame({
        'Maturity': [t['maturity'] for t in priced],
        'Coupon': [t['coupon'] for t in priced],
        'Amount ($)': [t['amount'] for t in priced],
        'Priced To': [t['ptc_date'] or 'Maturity' for t in priced],
        'Wire Yield': [t['yield'] for t in priced],
        'Wire Price': [t['price'] for t in priced],
        'Model Yield': np.round(pred, 4),
    }).set_index('Maturity')
    out['Error (bps)'] = ((out['Model Yield'] - out['Wire Yield']) * 100).round(1)
    # decomposition: total error = deal LEVEL (this deal's concession vs the
    # calibrated estimate -- irreducible ~+/-2bp until the deal count grows)
    # + tranche SHAPE (relative value within the deal, the model's strength).
    # Deal-Rel = error minus the deal's long-end mean level: which tranches
    # are rich/cheap RELATIVE TO THIS DEAL'S OWN PRICING LEVEL.
    long_end = ramp >= 0.999
    if long_end.sum() >= 3:
        level = float(out['Error (bps)'].to_numpy()[long_end].mean())
        out['Deal-Rel (bps)'] = (out['Error (bps)'] - level).round(1)
    # confidence: disagreement between ensemble members (high std = off the
    # training map), AND tranches under ~1.25yr -- the model trains on no
    # sub-1yr bonds (see train_yield_model), so its front-end reads are
    # extrapolation and get flagged rather than oversold.
    out['Model Std (bps)'] = np.round(ens_std_bps, 1)
    tenor_days = pd.to_numeric(fdf['days_to_maturity_30_360'], errors='coerce').to_numpy()
    out['Confidence'] = np.where((ens_std_bps <= 5) & (tenor_days >= 450), 'high', 'CHECK')

    err = out['Error (bps)'].abs()
    print(f"wire vs model: mean abs error {err.mean():.1f} bps | "
          f"median {err.median():.1f} | max {err.max():.1f}  "
          f"(concession {concession_bps:+.1f} bps)")
    return out
