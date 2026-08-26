"""
spline_pull -- fetch the latest Spline Data snapshot (bond-level pricing +
yield curves) and save both as CSVs next to this script.

The SFTP password is NOT hardcoded (it lives in plaintext in the old
Spline notebooks -- rotate it when you can). Supply it via env var:

    $env:SPLINE_PASSWORD = '<password from Spline 3.ipynb>'
    python spline_pull.py

Outputs:
    spline_pricing.csv  -- per-CUSIP bid/ask price+yield at 100k/500k/1mm
    spline_curves.csv   -- yield curves by rating x go/rev x liquidity x tenor
Both also archived with a date stamp in evals_archive\\.
"""
import io
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import paramiko

HERE = Path(__file__).resolve().parent
HOST = os.environ.get('SPLINE_HOST', 'sftp.splinedata.com')
PORT = int(os.environ.get('SPLINE_PORT', '22'))
USER = os.environ.get('SPLINE_USER', 'elequin2')
PWD = os.environ.get('SPLINE_PASSWORD')
if not PWD:
    sys.exit("Set $env:SPLINE_PASSWORD first (see Spline 3.ipynb for the value).")

SNAP_MIN, LOOKBACK_MIN = 5, 90


def round_down(t):
    return t.replace(minute=(t.minute // SNAP_MIN) * SNAP_MIN, second=0, microsecond=0)


def walk_back():
    ts = round_down(datetime.now())
    floor = ts - timedelta(minutes=LOOKBACK_MIN)
    while ts >= floor:
        yield ts
        ts -= timedelta(minutes=SNAP_MIN)


def read_parquet(sftp, path):
    with sftp.open(path, 'rb') as fh:
        fh.prefetch()
        return pd.read_parquet(io.BytesIO(fh.read()))


transport = paramiko.Transport((HOST, PORT))
transport.banner_timeout = 30
transport.connect(username=USER, password=PWD)
sftp = paramiko.SFTPClient.from_transport(transport)
try:
    pricing = curves = None
    for ts in walk_back():
        try:
            pricing = read_parquet(sftp, f"/pricing/{ts:%Y/%m/%d/%H/%M}/predictions.parquet")
            print(f"pricing snapshot {ts:%Y-%m-%d %H:%M}: {len(pricing):,} rows")
            break
        except IOError:
            continue
    for ts in walk_back():
        d = f"/curves/{ts:%Y/%m/%d/%H/%M}"
        try:
            names = sftp.listdir(d)
        except IOError:
            continue
        frames = []
        for name in names:
            try:
                c = read_parquet(sftp, f"{d}/{name}/curve.parquet")
                c.insert(0, 'curve', name)
                frames.append(c)
            except IOError:
                continue
        if frames:
            curves = pd.concat(frames, ignore_index=True)
            print(f"curves snapshot {ts:%Y-%m-%d %H:%M}: {len(curves):,} rows, "
                  f"{curves['curve'].nunique()} curves")
            break
finally:
    sftp.close()
    transport.close()

if pricing is None and curves is None:
    sys.exit(f"no snapshots found in the last {LOOKBACK_MIN} minutes -- market closed? "
             f"Try again during trading hours, or use the old notebook's 4pm fallback.")

stamp = datetime.now().strftime('%Y-%m-%d')
arch = HERE / 'evals_archive'
arch.mkdir(exist_ok=True)
if pricing is not None:
    pricing.to_csv(HERE / 'spline_pricing.csv', index=False)
    pricing.to_csv(arch / f'spline_pricing_{stamp}.csv', index=False)
    print(f"-> spline_pricing.csv (+ archived)")
if curves is not None:
    curves.to_csv(HERE / 'spline_curves.csv', index=False)
    curves.to_csv(arch / f'spline_curves_{stamp}.csv', index=False)
    print(f"-> spline_curves.csv (+ archived)")
print("DONE")
