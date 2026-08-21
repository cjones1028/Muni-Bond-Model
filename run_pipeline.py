"""
run_pipeline -- the automated path: Bloomberg wire text in, priced deal out.

    python run_pipeline.py --wire "BAML Write-Up.txt" --evals ICE_Evals.csv

What it does:
  1. trains the yield model from the evals CSV (or loads model.joblib if it
     exists -- pass --retrain to force a rebuild after a fresh eval pull)
  2. parses the wire (Wire_Parser)
  3. prices every tranche, prints wire-vs-model error in bps
  4. archives the result to wire_archive/<deal>_<timestamp>.csv -- the
     accumulating scoreboard for grinding error down below ICE's ~5bp

Options:
  --state "California"        train on one state's bonds (default: all)
  --issuer "LOS ANG"          issuer text for the categorical template
  --concession 3              new-issue concession in bps to add to model yield
  --retrain                   rebuild the model even if model.joblib exists
  --report                    print the bucketed holdout error report
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from Wire_Parser import parse_wire  # noqa: E402
import muni_model as mm  # noqa: E402


def auto_issuer(description, df, min_frac=0.5, min_hits=3):
    """Match the wire's issuer description to an ICE issuer name without a
    hand-typed --issuer. ICE names are abbreviated ('LOS ANG CY CA MET TRA
    AUT'), so score each name by the fraction of its tokens that are prefixes
    of words in the description. Returns the best name or None."""
    import re
    if 'primary_name_abbreviated' not in df.columns:
        return None
    words = {w for w in re.findall(r'[A-Za-z]{2,}', description.upper())}
    best_name, best_key = None, (0, 0.0, 0)
    counts = df['primary_name_abbreviated'].astype(str).str.upper().value_counts()
    for name, n_bonds in counts.items():
        toks = [t for t in name.split() if len(t) >= 2]
        if len(toks) < 3:
            continue
        # primary score: how many DISTINCT wire words this name accounts for
        # (prevents 'METROPOLITAN ... AUTH' of the wrong state winning on its
        # own short token list); frac = share of the name's tokens matched
        covered = {w for w in words if any(w.startswith(t) for t in toks)}
        hits = sum(any(w.startswith(t) for w in words) for t in toks)
        frac = hits / len(toks)
        key = (len(covered), frac, n_bonds)
        if frac >= min_frac and hits >= min_hits and key > best_key:
            best_name, best_key = name, key
    if best_name:
        print(f"auto-matched issuer: '{best_name}' ({best_key[2]:,} bonds, "
              f"{best_key[0]} wire words covered, {best_key[1]:.0%} token match)")
    else:
        print("no confident issuer match -- using full-universe template "
              "(pass --issuer to override)")
    return best_name


def default_concession():
    """Self-calibrating concession: mean implied concession across archived
    deals (see concession_tracker.py); falls back to 14 with no archive."""
    import pandas as pd
    import re as _re
    # one vote per DEAL: re-pricing the same wire creates multiple archive
    # files, so keep only the newest file per deal tag
    latest = {}
    for f in (HERE / 'wire_archive').glob('*.csv'):
        tag = _re.sub(r'_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', '', f.stem)
        if tag not in latest or f.stat().st_mtime > latest[tag].stat().st_mtime:
            latest[tag] = f
    vals = []
    for f in latest.values():
        try:
            d = pd.read_csv(f)
            if 'Error (bps)' in d.columns and 'Concession Used (bps)' in d.columns and len(d):
                vals.append(float(d['Concession Used (bps)'].iloc[0]) - d['Error (bps)'].mean())
        except Exception:
            continue
    if vals:
        c = round(sum(vals) / len(vals), 1)
        print(f"concession auto-calibrated: {c} bps from {len(vals)} archived deal(s)")
        return c
    print("concession default: 14 bps (no archive yet)")
    return 14.0


def main():
    p = argparse.ArgumentParser(description="Price a new-issue wire against the model.")
    p.add_argument('--wire', required=True, help='path to the wire text file')
    p.add_argument('--evals', default=str(HERE / 'ICE_Evals.csv'))
    p.add_argument('--model', default=str(HERE / 'model.joblib'))
    p.add_argument('--state', default=None, help="e.g. 'California' (default: all states)")
    p.add_argument('--issuer', default=None, help='issuer text for the categorical template')
    p.add_argument('--template-cusip', default=None, help='use this CUSIP as the template instead')
    p.add_argument('--concession', type=float, default=None,
                   help='new-issue concession, bps (default: auto-calibrated from wire_archive)')
    p.add_argument('--settlement', default=None, help="settlement date m/d/yyyy if the wire has no DATED line")
    p.add_argument('--retrain', action='store_true')
    p.add_argument('--report', action='store_true')
    p.add_argument('--no-archive', action='store_true',
                   help='skip writing to wire_archive (for tests/dry runs)')
    args = p.parse_args()
    if args.concession is None:
        args.concession = default_concession()

    # ----- model: load cache or train -----
    model_path = Path(args.model)
    cache_path = model_path.with_name('template_cache.parquet')
    df = None
    if model_path.exists() and not args.retrain:
        print(f"loading cached model {model_path.name} (use --retrain after a new eval pull)")
        bundle = mm.load_bundle(model_path)
    else:
        print(f"training from {args.evals} ...")
        df = mm.load_evals(args.evals)
        df = mm.clean_universe(df, state=args.state)
        bundle = mm.train_yield_model(df)
        if args.report:
            mm.error_report(bundle)
        mm.save_bundle(bundle, model_path)
        # slim per-bond cache so later pricing runs skip the big evals CSV
        tmpl_cols = [c for c in mm.CATEGORICAL_FEATURES
                     + ['primary_name_abbreviated'] if c in df.columns]
        df[tmpl_cols].to_parquet(cache_path)
        print(f"template cache -> {cache_path.name}")

    # ----- parse the wire first (issuer auto-match needs the description) -----
    text = open(args.wire, encoding='utf-8').read()
    deal = parse_wire(text)

    # ----- template categoricals (from slim cache when available) -----
    if df is None:
        if cache_path.exists():
            import pandas as pd
            df = pd.read_parquet(cache_path)
        else:
            df = mm.load_evals(args.evals)
            df = mm.clean_universe(df, state=args.state)
    issuer = args.issuer or auto_issuer(deal['description'], df)
    template = mm.template_from(df, issuer_contains=issuer, cusip=args.template_cusip)

    # credit-consistency guard: if the wire's rating is far from the matched
    # issuer's typical rating, this deal is a DIFFERENT credit of that issuer
    # (e.g. a BBB toll deal from an issuer whose outstanding bonds are AA
    # sales-tax). Keep the issuer's state/sector context but drop the
    # issuer-identity anchors and rating fields, which would otherwise pull
    # every tranche toward the wrong credit.
    import pandas as _pd
    tmpl_moody = _pd.to_numeric(_pd.Series([template.get('normalized_moody_long_rating')]),
                                errors='coerce').iloc[0]
    if (deal.get('moody_rating') is not None and _pd.notna(tmpl_moody)
            and abs(deal['moody_rating'] - tmpl_moody) >= 3 and not args.template_cusip):
        # a new credit has no true comparable in the issuer's own book; the
        # honest context is the LOCALITY (all issuers sharing the name's
        # geographic prefix) with the wire's rating layered on by price_wire.
        broad = ' '.join((issuer or '').split()[:2])
        print(f"credit mismatch (wire {deal['moody_rating']} vs pool {tmpl_moody:.0f}): "
              f"new credit of this issuer -- broadening template to '{broad}' "
              f"(pass --template-cusip of a true comparable for precision)")
        template = mm.template_from(df, issuer_contains=broad)

    print()
    print(deal['description'])
    size = f"${deal['issue_amount']:,}" if deal['issue_amount'] else "size not found on wire"
    print(f"deal {size} | Moody's {deal['moody']} / S&P {deal['sandp']} "
          f"/ Fitch {deal['fitch']} | dated {deal['dated_date']} | "
          f"call {deal['call_date']} @ {deal['call_price']} | {len(deal['tranches'])} tranches\n")

    results = mm.price_wire(deal, bundle, template,
                            settlement_date=args.settlement,
                            concession_bps=args.concession)
    # stored so concession_tracker.py can back out each deal's implied
    # concession from the archive later
    results['Concession Used (bps)'] = args.concession
    print()
    print(results.to_string())

    # ----- archive -----
    if args.no_archive:
        print("\n(--no-archive: result not written to wire_archive)")
        return
    arch_dir = HERE / 'wire_archive'
    arch_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    deal_tag = ''.join(ch for ch in deal['description'][:40] if ch.isalnum() or ch in ' -').strip().replace(' ', '_')
    out = arch_dir / f"{deal_tag}_{stamp}.csv"
    results.to_csv(out)
    print(f"\narchived -> {out}")


if __name__ == '__main__':
    main()
