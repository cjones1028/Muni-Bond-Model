"""Level vs shape error on the pinned regression wires with the current model.
Used to recalibrate test_suite thresholds after market-level moves."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm  # noqa: E402
from Wire_Parser import parse_wire  # noqa: E402

df = mm.clean_universe(mm.load_evals(HERE / 'ICE_Evals.csv'))
bundle = mm.load_bundle(HERE / 'model.joblib')
for name, wf, iss in [('LA', 'BAML Write-Up.txt', 'LOS ANG'),
                      ('NYC', 'NYC_TFA_wire.txt', 'CITY TRANSITIONAL FIN')]:
    deal = parse_wire(open(HERE / wf, encoding='utf-8').read())
    t = mm.template_from(df, issuer_contains=iss)
    r = mm.price_wire(deal, bundle, t, concession_bps=13)
    e = r['Error (bps)']
    line = f'{name}: abs err mean {e.abs().mean():.1f} | signed mean {e.mean():+.1f}'
    dr = r.get('Deal-Rel (bps)')
    if dr is not None:
        line += f' | Deal-Rel abs mean {dr.abs().mean():.1f}'
    print('RESULT', line)
