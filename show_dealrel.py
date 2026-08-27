"""Deal-relative (shape) accuracy on all four wires."""
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm
from Wire_Parser import parse_wire
from calibration import concession

WIRES = {'BAML Write-Up.txt': 'LOS ANG',
         'NYC_TFA_wire.txt': 'CITY TRANSITIONAL FIN',
         'psu_wire.txt': 'PENNSYLVANIA STATE UNIVERSITY',
         'portland_wire.txt': 'PORTLAND ORE SWR'}

tdf = pd.read_parquet(HERE / 'template_cache.parquet')
b = mm.load_bundle(HERE / 'model.joblib')
c = concession(verbose=False)
for w, iss in WIRES.items():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        deal = parse_wire(open(HERE / w, encoding='utf-8').read())
        t = mm.template_from(tdf, issuer_contains=iss)
        r = mm.price_wire(deal, b, t, concession_bps=c)
    if 'Deal-Rel (bps)' in r:
        print(f"{w}: total {r['Error (bps)'].abs().mean():.1f} -> "
              f"deal-relative {r['Deal-Rel (bps)'].abs().mean():.1f} "
              f"(median {r['Deal-Rel (bps)'].abs().median():.1f})")
    else:
        print(f"{w}: too few long-end tranches for decomposition")
