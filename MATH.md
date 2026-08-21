# Model Mathematics — verification reference

Every formula as implemented, each with a hand-checkable example from the
live output (top row of `deal_ranking.csv`: LA Metro 2051 5.50%).

## 1. Day count — 30/360 US (`muni_model.days_30_360`)

    d1 = min(day1, 30)
    d2 = 30 if (day2 == 31 and d1 == 30) else day2
    days = 360*(y2-y1) + 30*(m2-m1) + (d2-d1)

Check: 9/3/2026 -> 6/1/2036 = 360*10 + 30*(6-9) + (1-3) = 3,508.

## 2. Yield model (`muni_model.train_yield_model`)

Ensemble of 3 LightGBM gradient-boosted tree models, objective = L1
(minimize sum |y - F(x)|), 8,000 trees, learning rate 0.03, early stopping
on an 8% validation slice. Target y = ICE yield_mid.

    y_hat = mean_k f_k(x) + concession/100        (concession = 14 bps)
    error_bps = (y_hat - y) * 100
    MAE = mean |error_bps| ; also median

## 3. Concession (`concession_tracker.py`)

    implied_concession(deal) = concession_used - mean(signed error of deal)
    estimate = mean over archived deals

Check: NYC ran at 13 with mean signed error -1.2 -> implied 14.2.
Two deals: (13.6 + 14.2)/2 = 14.

## 4. Bond price / DV01 (`rank_deals.py`)

Street semiannual PV to workout (call date if priced-to-call, else maturity),
n = round(2 * years):

    P(y) = sum_{t=1..n} (cpn/2)/(1+y/2)^t + 100/(1+y/2)^n
    DV01 = [P(y-0.005) - P(y+0.005)] / 0.01     (price points per 1% yield)
    $ per $1mm face = bps/100 * DV01 * 10,000

Check: 2051, cpn 5.50, 9.75yr to call, y=4.92 -> DV01 = 7.71.

## 5. Risk (uncertainty budget)

    sigma_total = sqrt(sigma_bucket^2 + sigma_ens^2 + sigma_conc^2)

    sigma_bucket = 1.25 * holdout MAE of the tranche's coupon x tenor bucket
                   (1.25 ~ sqrt(pi/2), MAE->stdev under normality;
                    table cached in bucket_errors.csv, regen after retrain)
    sigma_ens    = stdev of the 3 members' predictions * 100
    sigma_conc   = 3  (concession estimated from only 2 deals)

Check: sqrt(6.1^2 + 2.9^2 + 3.0^2) = sqrt(54.6) = 7.4.

## 6. Edge, probability, profit score

    gap   = (wire yield - model yield) * 100      (+ = screens cheap)
    net   = gap - exit_cost                        (exit_cost = 5 bps)
    ratio = net / sigma_total
    P(real) = Phi(ratio) = 0.5 * (1 + erf(ratio / sqrt(2)))

    PROFIT $/mm = (net - 0.5 * sigma_total)/100 * DV01 * 10,000

Check (2051 row): gap 8.8 -> net 3.8; ratio 3.8/7.4 = 0.51; Phi = 70%;
EV = 3.8/100 * 7.71 * 10,000 = $2,930;
PROFIT = (3.8 - 3.7)/100 * 7.71 * 10,000 ~ $80.

## 7. Derived model features (`muni_model.build_features`)

    log_issue_amount        = ln(1 + issue_amount)
    log_outstanding_amount  = ln(1 + outstanding_amount)
    outstanding_pct_of_issue= outstanding / issue
    call_moneyness          = mid - next_call_price   (0 if non-callable)
    call_moneyness_pct      = call_moneyness / next_call_price
    above_call_price        = 1 if mid > next_call_price else 0
    days_to_refund_30_360   = refund_date - snapshot date (rebuilt from feed)

## Conventions vs. mathematics

Three constants are policy choices, not derivations:
  exit cost 5 bps | risk charge 0.5 sigma | sigma_conc 3 bps.
Everything else is measured (bucket errors, ensemble spread, implied
concession) or standard fixed-income math (30/360, semiannual PV, DV01).
Adjust the three knobs at the top of rank_deals.py to express a different
risk appetite; do not adjust the measured components.
