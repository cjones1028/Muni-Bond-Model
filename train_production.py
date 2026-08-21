"""Train the production ensemble (current muni_model defaults) and save
model.joblib + template_cache.parquet. Writes train_production.done with the
holdout metrics when finished -- used by unattended/detached runs."""
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm

DONE = HERE / 'train_production.done'
DONE.unlink(missing_ok=True)

try:
    df = mm.load_evals(HERE / 'ICE_Evals.csv')
    df = mm.clean_universe(df)
    bundle = mm.train_yield_model(df)
    mm.save_bundle(bundle, HERE / 'model.joblib')
    cols = [c for c in mm.CATEGORICAL_FEATURES + ['primary_name_abbreviated'] if c in df.columns]
    df[cols].to_parquet(HERE / 'template_cache.parquet')
    DONE.write_text(f"OK mae={bundle['mae_bps']:.2f} median={bundle['median_bps']:.2f}\n")
    print("done")
except Exception:
    DONE.write_text("FAILED\n" + traceback.format_exc())
    raise
