"""Generate MuniPricer.ipynb -- the Jupyter front-end for the pipeline."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    md("# Muni New-Issue Pricer\n"
       "Paste-a-wire -> priced deal. See README.md for the full manual, "
       "MATH.md for every formula.\n\n"
       "**First time on this machine:** run the *Setup* cell, then *Retrain* "
       "(~20 min, once per data refresh). After that, only the *Price a wire* "
       "cell is needed day to day."),

    md("## Setup"),
    code("import sys, pathlib\n"
         "sys.path.insert(0, str(pathlib.Path().resolve()))\n"
         "import muni_model as mm\n"
         "from Wire_Parser import parse_wire\n"
         "print('imports ok')"),

    md("## (One-time / after each data refresh) Pull data + retrain\n"
       "Data: run `Pull_ICE_Evals.ps1` in PowerShell first (needs the API "
       "key -- see README). Then retrain here:"),
    code("# ~20 minutes; writes model.joblib + template_cache.parquet\n"
         "# %run train_production.py"),

    md("## Price a wire\n"
       "Save the Bloomberg message as a .txt in this folder, put its name "
       "below, run the cell."),
    code("WIRE = 'BAML Write-Up.txt'   # <-- your saved wire file\n"
         "\n"
         "%run run_pipeline.py --wire \"$WIRE\""),

    md("## Risk-adjusted ranking (PROFIT $/mm per tranche)\n"
       "Add new deals to the DEALS list inside rank_deals.py, then:"),
    code("%run rank_deals.py"),

    md("## Concession tracker (what discount are dealers pricing at?)"),
    code("%run concession_tracker.py"),

    md("## Screen the whole secondary universe (~2 min)"),
    code("# %run screen_universe.py"),

    md("## Verify everything (run after ANY code change)"),
    code("import os\n"
         "os.environ['TEST_REUSE_MODEL'] = '1'   # validate existing model, no retrain\n"
         "%run test_suite.py"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

out = HERE / "MuniPricer.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}")
