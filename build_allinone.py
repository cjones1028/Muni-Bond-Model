"""Build MuniPricer_AllInOne.ipynb -- a single notebook containing the whole
system. Each module is a %%writefile cell, so running the notebook top to
bottom recreates the .py files locally and then uses them. Uploads anywhere
a notebook uploads (Anaconda Cloud, JupyterLab, Colab...)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODULES = ['muni_model.py', 'Wire_Parser.py', 'run_pipeline.py',
           'train_production.py', 'concession_tracker.py', 'test_suite.py']


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    md("# Muni New-Issue Pricer — all-in-one\n"
       "This single notebook contains the complete system. **Run the cells in "
       "order once**: they install dependencies and write the code modules to "
       "disk next to this notebook. After that, only the *Price a wire* "
       "section is needed day to day.\n\n"
       "You also need `ICE_Evals.csv` (the training data) in the same folder — "
       "upload it alongside this notebook, or produce it with the ICE pull "
       "described inside `run_pipeline.py`'s docs."),
    md("## 1. Dependencies (once per environment)"),
    code("%pip install -q pandas==3.0.5 numpy scikit-learn==1.9.0 "
         "lightgbm==4.7.0 joblib pyarrow"),
    md("## 2. Write the code modules (once, or after edits here)"),
]

for name in MODULES:
    src = (HERE / name).read_text(encoding='utf-8')
    cells.append(code(f"%%writefile {name}\n{src}"))

cells += [
    md("## 3. Train the model (once per data refresh, ~20 min)\n"
       "Needs `ICE_Evals.csv` in this folder. Writes `model.joblib` and "
       "`template_cache.parquet`."),
    code("# %run train_production.py"),

    md("## 4. Price a wire\n"
       "Paste the FULL Bloomberg wire message between the triple quotes and "
       "run. (Or skip this cell and run "
       "`%run run_pipeline.py --wire yourfile.txt` on a saved file.)"),
    code('WIRE_TEXT = """\nPASTE THE ENTIRE WIRE MESSAGE HERE\n"""\n'
         "\n"
         "open('pasted_wire.txt', 'w', encoding='utf-8').write(WIRE_TEXT)\n"
         "%run run_pipeline.py --wire pasted_wire.txt"),

    md("## 5. Concession tracker (run occasionally)"),
    code("%run concession_tracker.py"),

    md("## 6. Verify the installation / any code change"),
    code("import os\n"
         "os.environ['TEST_REUSE_MODEL'] = '1'\n"
         "%run test_suite.py"),
]

import sys as _sys
# kernel name can be passed on the command line to match a specific
# environment, e.g.  python build_allinone.py anaconda-2025.12-py312
KERNEL = _sys.argv[1] if len(_sys.argv) > 1 else "python3"
DISPLAY = KERNEL if KERNEL != "python3" else "Python 3"
SUFFIX = "" if KERNEL == "python3" else "_anaconda"

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": DISPLAY, "language": "python",
                                  "name": KERNEL},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

out = HERE / f"MuniPricer_AllInOne{SUFFIX}.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(cells)} cells, kernel '{KERNEL}')")
