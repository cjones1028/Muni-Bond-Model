"""Does more training capacity help? Single model, same split: 3000-tree cap
(current) vs 8000 with early stopping. Judged on holdout only -- the two
wires are a frozen regression set, not a tuning target."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muni_model as mm

df = mm.load_evals(HERE / 'ICE_Evals.csv')
df = mm.clean_universe(df)

for cap in (3000, 8000):
    b = mm.train_yield_model(df, seed=42, n_ensemble=1, n_estimators=cap,
                             learning_rate=0.03)
    print(f">>> cap {cap}: MAE {b['mae_bps']:.2f} | median {b['median_bps']:.2f} "
          f"| best_iter {b['models'][0].best_iteration_}")
