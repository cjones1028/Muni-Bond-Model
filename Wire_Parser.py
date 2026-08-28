"""
Wire_Parser -- parse a dealer new-issue pricing wire (the standard format in
'BAML Write-Up.txt') and run EVERY tranche through the ICE / Spline yield and
DV01 models, comparing model yield to the dealer's yield in bps.

This replaces the hand-typed inputs of 'Run Code Part 2.txt'. The wire format
this expects:

    RE: $ 509,145,000*
    <issuer description lines>
    MOODY'S: Baa2 (Stable)                  S&P:   NR
    FITCH:   BBB- (Stable)                  KROLL: NR
    DATED:09/03/2026   FIRST COUPON:12/01/2026
    06/01/2034      5,650M     5.00%     3.57      0.40
                          (Approx. $ Price 109.595)
    06/01/2037      8,600M     5.00%     3.90      0.40
                          (Approx. $ Price PTC 06/01/2036 108.840)
    ...
    CALL FEATURES:  Optional call in 06/01/2036 @ 100.00

Usage (see 'Run Code Part 3.txt'; requires the models from Run Code Part 1
plus predict_curve / days_30_360 already in the namespace):

    deal    = parse_wire(open(path).read())
    results = price_wire_tranches(deal, df,
                                  ice_yield_model, spline_yield_model,
                                  ice_dv01_model, spline_dv01_model)

parse_wire() is pure stdlib text -> dict, so it is trivially testable on any
new wire before the models ever get involved.
"""

import re
from datetime import datetime

import pandas as pd

# Numeric scales matching ratings_dict in ICE_Data_Pull (1 = AAA ... 22 = D).
MOODY_SCALE = {'Aaa': 1, 'Aa1': 2, 'Aa2': 3, 'Aa3': 4, 'A1': 5, 'A2': 6, 'A3': 7,
               'Baa1': 8, 'Baa2': 9, 'Baa3': 10, 'Ba1': 11, 'Ba2': 12, 'Ba3': 13,
               'B1': 14, 'B2': 15, 'B3': 16, 'Caa1': 17, 'Caa2': 18, 'Caa3': 19,
               'Ca': 20, 'C': 21}
SP_SCALE = {'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4, 'A+': 5, 'A': 6, 'A-': 7,
            'BBB+': 8, 'BBB': 9, 'BBB-': 10, 'BB+': 11, 'BB': 12, 'BB-': 13,
            'B+': 14, 'B': 15, 'B-': 16, 'CCC+': 17, 'CCC': 18, 'CCC-': 19,
            'CC': 20, 'C': 21, 'D': 22}

# maturity row, tolerant of common dialect variations:
#   - optional 1-2 letter flag after the date (sinking fund 'S', term 'T')
#   - optional "orders / +spread" token (Loop Capital: "78,475 / +40")
#   - optional '+' after the yield (priced-to-premium-call stub)
#   - trailing extras (same-line price parenthetical, ratings text)
# The specific anchors (date + amount'M' + coupon'%' + yield) keep narrative
# lines from matching; the tranche-sum check catches anything that slips.
MATURITY_ROW = re.compile(
    r'^\s*(\d{2}/\d{2}/\d{4})(?:\s+[A-Z]{1,2}(?=\s))?\s+(?:[\d,]+M?\s*/\s*\+?-?\d+\s+)?'
    r'([\d,]+)M\s+([\d.]+)%\s+([\d.]+)\+?(?:\s+([\d.]+))?(?:\s+\S.*)?$')
# note the M? in the orders token: Wells Fargo prints "2,500M / +7" (amount
# with M before the slash), Loop prints "78,475 / +40" (without)
PRICE_ROW = re.compile(
    r'\(Approx\.\s*\$\s*Price\s*(?:PTC\s*(\d{2}/\d{2}/\d{4})\s*)?([\d.]+)')
# no closing \) required: Wells Fargo wraps price lines mid-parenthesis
# ("...107.266 Approx.\nYTM 4.387)") -- the price is captured before the wrap


def _date(s):
    return datetime.strptime(s, '%m/%d/%Y')


def _rating(text, agency):
    m = re.search(re.escape(agency) + r"\s*:\s*(\S+)", text)
    if not m:
        return 'NR'
    raw = m.group(1)
    # a blank rating slot makes \S+ swallow the NEXT label (seen live:
    # "FITCH:            KROLL: NR" parsed Fitch as "KROLL:")
    if raw.endswith(':'):
        return 'NR'
    # real wires carry compound formats -- "Aa1/VMIG-1" (long/short dual
    # rating), "AA+*" (watch flag). Take the long-term component and strip
    # decorations before validating against the agency scales.
    val = raw.split('/')[0].rstrip('*').strip()
    if val in MOODY_SCALE or val in SP_SCALE or val.upper() == 'NR':
        return val
    print(f"NOTE: unrecognized {agency} rating '{raw}' on this wire -- "
          f"treating as NR (template rating will be used)")
    return 'NR'


def parse_wire(text):
    """Parse one pricing wire into deal-level fields + a list of tranches."""
    deal = {}
    lines = text.splitlines()

    m = re.search(r'RE:\s*\$\s*([\d,]+)', text)
    if not m:
        # no "RE: $" header (some dealers put the size in the Subject line) --
        # fall back to the first large dollar amount anywhere in the text
        m = re.search(r'\$\s*([\d,]{9,})', text)
    deal['issue_amount'] = int(m.group(1).replace(',', '')) if m else None

    # issuer description: the non-blank lines following the RE: line
    desc = []
    for j, line in enumerate(lines):
        if 'RE:' in line:
            for nxt in lines[j + 1:]:
                s = nxt.strip()
                if not s:
                    if desc:
                        break
                    continue
                desc.append(s)
            break
    deal['description'] = ' '.join(desc)
    if not deal['description']:
        m = re.search(r'Subject\s+(.+)', text)
        deal['description'] = m.group(1).strip() if m else 'UNKNOWN DEAL'

    deal['moody'] = _rating(text, "MOODY'S")
    deal['sandp'] = _rating(text, 'S&P')
    deal['fitch'] = _rating(text, 'FITCH')

    # Numeric ratings for the model; if an agency is NR fall back to the
    # others so the model always gets a number (S&P NR -> use Fitch, etc.)
    moody_num = MOODY_SCALE.get(deal['moody'])
    sandp_num = SP_SCALE.get(deal['sandp'])
    fitch_num = SP_SCALE.get(deal['fitch'])
    deal['moody_rating'] = moody_num if moody_num is not None else (sandp_num or fitch_num)
    deal['sandp_rating'] = sandp_num if sandp_num is not None else (fitch_num or moody_num)

    m = re.search(r'DATED\s*:\s*(\d{2}/\d{2}/\d{4})', text)
    deal['dated_date'] = m.group(1) if m else None
    m = re.search(r'FIRST\s+COUPON\s*:\s*(\d{2}/\d{2}/\d{4})', text)
    deal['first_coupon'] = m.group(1) if m else None

    m = re.search(r'call\s+in\s+(\d{2}/\d{2}/\d{4})\s*@\s*([\d.]+)', text, re.IGNORECASE)
    deal['call_date'] = m.group(1) if m else None
    deal['call_price'] = float(m.group(2)) if m else 100.0

    tranches = []
    for i, line in enumerate(lines):
        m = MATURITY_ROW.match(line)
        if not m:
            continue
        tranche = {
            'maturity': m.group(1),
            'amount': int(m.group(2).replace(',', '')) * 1000,  # wire prints $ thousands
            'coupon': float(m.group(3)),
            'yield': float(m.group(4)),
            'takedown': float(m.group(5)) if m.group(5) else None,
            'price': None,
            'ptc_date': None,
        }
        # the (Approx. $ Price ...) is usually on a following line, but some
        # dialects put it at the end of the maturity row itself
        for nxt in lines[i:i + 3]:
            pm = PRICE_ROW.search(nxt)
            if pm:
                tranche['ptc_date'] = pm.group(1)
                tranche['price'] = float(pm.group(2))
                break
        tranches.append(tranche)
    deal['tranches'] = tranches

    total = sum(t['amount'] for t in tranches)
    if deal['issue_amount'] and total != deal['issue_amount']:
        print(f"WARNING: tranche amounts sum to {total:,} "
              f"vs stated issue size {deal['issue_amount']:,} -- check the parse.")
    return deal


def price_wire_tranches(deal, df,
                        ice_yield_model, spline_yield_model,
                        ice_dv01_model, spline_dv01_model,
                        settlement_date=None, template_cusip=None,
                        output_path=None):
    """
    Run every tranche of a parsed wire through the four models. Returns a
    DataFrame with model yields vs the dealer's wire yields, error in bps,
    and predicted DV01s. Prints a mean/median/max abs-error summary -- the
    number to beat is ICE's ~5bp.

    df and the four models are the outputs of build_ice_spline_curves.
    Needs predict_curve and days_30_360 in the namespace (%run the
    Curve_Analysis / New_Holidays notebooks first).

    template_cusip: existing bond of this issuer whose categorical features
    (state, sector, security type...) stand in for the new issue. Defaults
    to df.index[0] -- fine while df is filtered to one issuer, but pass it
    explicitly if the filter ever widens.
    """
    settlement = settlement_date or deal['dated_date']
    call_date, call_price = deal['call_date'], deal['call_price']
    cusip = template_cusip if template_cusip is not None else df.index[0]

    rows = []
    for t in deal['tranches']:
        if t['price'] is None:
            print(f"Skipping {t['maturity']} {t['coupon']}% -- no dollar price parsed.")
            continue

        days_to_maturity = days_30_360(settlement, t['maturity'])

        # callable only if it matures AFTER the call date (the wire's PTC tag
        # marks priced-to-call, not callability -- a discount tranche past the
        # call date is still callable but priced to maturity)
        callable_ = call_date is not None and _date(t['maturity']) > _date(call_date)
        if callable_:
            days_to_call = days_30_360(settlement, call_date)
            days_call_to_mat = days_to_maturity - days_to_call
            call_moneyness = t['price'] - call_price
            call_moneyness_pct = call_moneyness / call_price
            above_call = float(t['price'] > call_price)
            next_call = call_price
        else:
            days_to_call = ''
            days_call_to_mat = ''
            call_moneyness, call_moneyness_pct, above_call = 0.0, 0.0, 0.0
            next_call = 100.0

        base = {
            'current_coupon_rate': t['coupon'],
            'call_indicator': str(callable_),
            'days_to_maturity_30_360': days_to_maturity,
            'days_to_next_call_30_360': days_to_call,
            'days_to_first_call_30_360': days_to_call,
            'days_from_call_to_maturity_30_360': days_call_to_mat,
            'issue_amount': deal['issue_amount'],
            'outstanding_amount': t['amount'],
            'next_call_price': next_call,
        }

        # NOTE: the DV01 feature lists use 'current_coupon' (not
        # 'current_coupon_rate'), and the Spline DV01 model uses 'Spline Mid'
        # (not 'mid') -- Run Code Part 2 set the wrong keys, so the coupon and
        # price never reached those models. Correct per-model keys here.
        ice_dv01_changes = {**base, 'current_coupon': t['coupon'],
                            'mid': t['price'],
                            'ICE call_moneyness': call_moneyness,
                            'ICE call_moneyness_pct': call_moneyness_pct,
                            'ICE above_call_price': above_call}
        spline_dv01_changes = {**base, 'current_coupon': t['coupon'],
                               'Spline Mid': t['price'],
                               'Spline call_moneyness': call_moneyness,
                               'Spline call_moneyness_pct': call_moneyness_pct,
                               'Spline above_call_price': above_call}

        ice_dv01 = predict_curve(model=ice_dv01_model[0], result=ice_dv01_model[1],
                                 cusip=cusip, **ice_dv01_changes)
        spline_dv01 = predict_curve(model=spline_dv01_model[0], result=spline_dv01_model[1],
                                    cusip=cusip, **spline_dv01_changes)

        yield_base = {**base,
                      'normalized_moody_long_rating': deal['moody_rating'],
                      'normalized_sandp_long_rating': deal['sandp_rating']}
        ice_yield_changes = {**yield_base,
                             'ICE call_moneyness': call_moneyness,
                             'ICE call_moneyness_pct': call_moneyness_pct,
                             'ICE above_call_price': above_call}
        spline_yield_changes = {**yield_base,
                                'Spline call_moneyness': call_moneyness,
                                'Spline call_moneyness_pct': call_moneyness_pct,
                                'Spline above_call_price': above_call}

        ice_yld = predict_curve(ice_yield_model[0], ice_yield_model[1],
                                cusip='', **ice_yield_changes)
        spline_yld = predict_curve(spline_yield_model[0], spline_yield_model[1],
                                   cusip='', **spline_yield_changes)

        rows.append({
            'Maturity': t['maturity'],
            'Coupon': t['coupon'],
            'Amount ($)': t['amount'],
            'Priced To': t['ptc_date'] or 'Maturity',
            'Wire Yield': t['yield'],
            'Wire Price': t['price'],
            'ICE Yield': round(ice_yld, 4),
            'Spline Yield': round(spline_yld, 4),
            'ICE Error (bps)': round((ice_yld - t['yield']) * 100, 1),
            'Spline Error (bps)': round((spline_yld - t['yield']) * 100, 1),
            'ICE DV01': round(ice_dv01, 4),
            'Spline DV01': round(spline_dv01, 4),
        })

    results = pd.DataFrame(rows).set_index('Maturity')

    for name in ('ICE', 'Spline'):
        err = results[f'{name} Error (bps)'].abs()
        print(f"{name:6s}: mean abs error {err.mean():.1f} bps | "
              f"median {err.median():.1f} | max {err.max():.1f}")

    if output_path:
        results.to_csv(output_path)
    return results
