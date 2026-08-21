import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm

df = mm.load_evals(HERE / 'ICE_Evals.csv')
df = mm.clean_universe(df)
cols = [c for c in mm.CATEGORICAL_FEATURES + ['primary_name_abbreviated'] if c in df.columns]
df[cols].to_parquet(HERE / 'template_cache.parquet')
print(f"template_cache.parquet: {len(df):,} rows, {len(cols)} cols")
