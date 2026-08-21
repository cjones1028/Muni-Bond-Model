# Muni New-Issue Pricing Model

Automated fair-value pricing of municipal new issues directly from dealer
pricing wires. Paste a wire off Bloomberg into a text file, run one command,
and every tranche is priced against the model with error reported in basis
points. No hand-typed inputs.

This is an upgraded replacement for the prior workflow (per-issuer linear
model + manually keyed bond attributes).

## Results to date

| Test | Old approach (linear) | This model |
|---|---|---|
| Old approach's designed use: one large issuer, refit per issuer (matched 5-fold CV, LA MTA pool) | ~4.0 bps | **3.3 bps** |
| One model pricing the whole universe, no per-issuer refit (16,482-bond holdout vs ICE marks) | 20.2 bps* | **5.3 bps mean / 2.2 median** |
| LA Metro I-105 toll deal, $509mm, 16 tranches (vs dealer wire, concession 14) | n/a (new credit, too few bonds to fit) | **2.5 bps mean** |
| NYC TFA, $1.5B, 7 tranches, out-of-sample (vs dealer wire, concession 14) | — | **1.0 bps mean (max 2.5)** |

The 14 bps concession comes from `concession_tracker.py`, which backs an
implied concession out of every archived deal (13.6 and 14.2 on the first
two) -- it re-estimates automatically as deals accumulate.

\* Not how the old approach was used in practice -- it was always refit per
issuer, where it scores ~4-5 bps. The 20.2 quantifies why it REQUIRED that
per-issuer refit: a single linear fit cannot cover the whole market, which is
what fully automated wire pricing needs. The new model keeps ~4-6 bps
accuracy universe-wide with one trained model. Head-to-head on identical
data/features/splits: the gradient-boosted model won 12/12 simulation rounds
and was closer on 85% of individual bonds (`model_comparison.csv`,
`fair_fight_results.csv`, `issuer_level_comparison.csv`).

Both wire tests used a fixed +13 bps new-issue concession (estimated on the
LA deal, applied blind to the NYC deal). Every run archives to
`wire_archive\`; that history refines the concession estimate over time.

## Daily use

```
python run_pipeline.py --wire "<saved wire>.txt" --issuer "<ICE issuer text>" --concession 13
```

- `--wire` (required): the dealer wire, copied in full from Bloomberg into a
  .txt file. Formats handled: BAML-style and Loop Capital-style; a new
  dealer's dialect may need a small parser extension (the built-in
  tranche-sum check flags any parse problem loudly).
- `--issuer`: text matching the issuer in ICE data (e.g. "CITY TRANSITIONAL
  FIN"). Supplies sector/state/tax-status context for the new bonds.
  IMPORTANT: match the CREDIT, not just the name -- for a deal that is a new
  or different credit of an existing issuer, use a broad match or
  `--template-cusip` of a comparable bond. The pipeline warns when the wire's
  rating is far from the template pool's.
- `--concession`: bps added to the model's secondary-market yield (new-issue
  discount). Current estimate: ~13 (BBB), ~10 (AAA).
- `--settlement m/d/yyyy`: only needed if the wire lacks a DATED line.
- `--retrain`: rebuild the model after refreshing ICE_Evals.csv.
- `--report`: print the bucketed holdout error report after training.

## Bloomberg testing protocol (per wire)

1. Bloomberg message open -> select all -> copy -> paste into Notepad ->
   save as `<dealer>_<deal>.txt` in this folder. Paste the WHOLE message.
2. Drag the .txt onto `Price Wire.bat` (or run run_pipeline.py --wire ...).
   The issuer is auto-matched from the wire text; pass --issuer only if the
   printed match looks wrong.
3. Read three things on the output:
   - the tranche-sum WARNING (fires = the parse missed rows; a new dealer
     dialect needs a parser extension -- keep the wire, flag it)
   - the Confidence column (any CHECK row = verify that tranche by hand)
   - mean abs error vs the wire, the accuracy scorecard for that deal
4. The run auto-archives. Periodically run `python concession_tracker.py`
   and update the concession in `Price Wire.bat` / rank_deals.py when the
   recommendation moves.
5. For the risk-adjusted PROFIT ranking add the deal to rank_deals.py's
   DEALS list and run it.

Every wire priced makes the concession estimate sharper and the accuracy
claim broader -- the archive is the system's report card, deal by deal.

## Refreshing data (weekly or before an important deal)

```
$env:ICE_ACCESS_KEY = '<key>'
powershell -ExecutionPolicy Bypass -File Pull_ICE_Evals.ps1 -InputCsv Universe_Clean.csv
python run_pipeline.py --wire <any wire> --retrain ...
```

## Files

| File | Purpose |
|---|---|
| `run_pipeline.py` | one-command driver: wire in, priced table out, archived |
| `Wire_Parser.py` | wire text -> structured deal (deal terms + all tranches) |
| `muni_model.py` | model: training, prediction, error reports, day counts |
| `test_suite.py` | 40+ verification checks -- run after any code change |
| `Pull_ICE_Evals.ps1` / `Pull_Universe.ps1` | data downloaders (Windows PowerShell) |
| `requirements.txt` | pinned Python environment (`pip install -r requirements.txt`) |
| `ICE_Evals.csv` | training data: ICE evals for 118,791 CUSIPs (refresh regularly) |
| `Universe_Clean.csv` | validated CUSIP universe |
| `model.joblib` | trained model cache (auto-rebuilt with `--retrain`) |
| `wire_archive\` | every priced deal, timestamped -- the accuracy scoreboard |
| `screen_universe.py` | rank the universe by model-vs-market gap, split into actionable (trusted tiers A/B) vs research-only (tier C); writes `screen_actionable.csv` / `screen_research.csv` |
| `compare_models.py`, `fair_fight.py`, `compare_issuer_level.py` | the old-vs-new simulations |
| `diag_confidence.py`, `diag_tail.py`, `exp_config.py` | the validation experiments behind the trust filter and feature choices |
| `smoke_test.py` | end-to-end plumbing check on synthetic data (no real data needed) |
| `ICE_Data_Pull.ipynb`, `ICE_Trading.ipynb`, `Spline.ipynb` | original notebooks (reference) |

## Model design (one paragraph)

A 3-seed ensemble of LightGBM gradient-boosted models (L1 objective -- it
optimizes the absolute-error metric we report -- with early stopping) trained
on the full tax-exempt fixed-rate universe
(~84,000 bonds after filters), predicting ICE mid yield from structure
(coupon, 30/360 day counts to maturity/call, call moneyness, sink/refund
fields, size), credit (Moody's/S&P normalized ratings, composite ratings,
insurance), and identity (issuer id, state, sector, security type, tax
status). New-issue tranches are priced by building the same feature vector
from the parsed wire, borrowing unobservable categoricals from the issuer's
existing bonds, and adding the concession. Categorical values are
canonicalized so wire-supplied inputs always match training categories.
Candidate features were accepted/rejected by experiment on real wires
(`exp_config.py`); five that helped nothing and hurt new-issue pricing are
documented as rejected in `muni_model.py`. A cached pricing run (parsed wire
-> full priced table) takes ~5 seconds; training after a data refresh takes a
few minutes and is cached in `model.joblib` + `template_cache.parquet`.

## Adverse selection -- READ THIS before using gaps as trade signals

The model's biggest misses are bonds where the MARKET knows something the
model's features can't see: distressed story credits marked at 8-18% yields,
escrowed/pre-refunded bonds trading at treasury levels, ultra-short and
micro issues. If you screen the universe for "cheapest vs model", those are
exactly what floats to the top -- a normal-looking gap can be a value trap,
not an opportunity.

The system defends against this two ways, both validated on holdout
(`diag_confidence.py`):
* every priced tranche carries a `Confidence` column driven by ensemble
  disagreement (the 3 models voting apart = off the training map);
* `confidence_flags()` applies the full trust filter (rating, conduit, size,
  tenor, yield level, disagreement). Trusted bonds (65% of universe): mean
  error 3.5 bps, median 1.7. The filter catches 89% of >50bp and 95% of
  >100bp model errors in advance. A big gap on a flagged bond is a research
  lead for a human, never an automatic signal.

## Known limitations

1. Wire parser covers two dealer dialects so far; new dialects need one-time
   extensions. The tranche-sum check catches silent misparses.
0. (Addressed) Pre-refunded/escrowed bonds are priced off `refund_date` and
   `called_redemption_type_desc` (features since 2026-08-19); dead instruments
   are filtered from training; the screener costs each bond by its own quoted
   bid/ask width. Screener gaps remain vs ICE marks -- the next capturability
   upgrades are Spline size-bucket pricing on screened names, then MSRB trade
   prints.
2. The 13 bps concession is a 2-deal estimate; it varies by credit and market
   tone. The archive exists to refine it.
3. Model is weakest where training data is thin: coupons above 6% (~25 bps)
   and maturities beyond 30 years (~11 bps). Core market (4-5% coupons,
   5-30yr) runs 4-6 bps.
4. Trained on a single day's eval snapshot; refresh + retrain keeps it
   current, and accumulating daily snapshots would enable time-based
   validation (stronger than the random splits used now).
5. Requires Python 3.12 with `requirements.txt` versions. On a new machine:
   copy this folder, `pip install -r requirements.txt`, run once with
   `--retrain` (do not copy model.joblib across machines).

## Verification

`python test_suite.py` runs the full audit: day-count math vs hand-computed
values, both wires parsed field-by-field against the source text, edge cases
(no call features, missing prices, at-call maturities), categorical
canonicalization, model determinism, save/reload identity, and end-to-end
pricing regression on both real deals. All checks pass as of 2026-08-19.
