"""Build Pricer.ipynb -- the daily-driver notebook that cannot go stale.
Every run shells out to a FRESH pinned-interpreter process, so kernel
caching, wrong environments, and stale modules are structurally impossible.
No %%writefile cells -- nothing in it can overwrite the code."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNNER = r'''# ============== PRICE A DEAL — the only cell you need ==============
# EITHER put a saved wire filename here...
WIRE = "portland_wire.txt"     # e.g. "BAML Write-Up.txt", NYC_TFA_wire.txt, psu_wire.txt, ca_wire.txt

# ...OR paste a NEW Bloomberg wire between the triple quotes (this then
# overrides WIRE above). Leave empty to use the file named above.
WIRE_TEXT = """
"""

# ---- no need to edit below: runs a FRESH process every time (never stale) ----
import subprocess, pathlib
PY = r"C:\Users\CharlieJones\AppData\Local\Programs\Python\Python312\python.exe"
folder = pathlib.Path(r"C:\Users\CharlieJones\Downloads\Codes")
if WIRE_TEXT.strip():
    (folder / "pasted_wire.txt").write_text(WIRE_TEXT, encoding="utf-8")
    WIRE = "pasted_wire.txt"
r = subprocess.run([PY, "run_pipeline.py", "--wire", WIRE],
                   capture_output=True, text=True, cwd=folder)
print(r.stdout)
if r.returncode != 0:
    print("--- ERROR ---")
    print(r.stderr[-3000:])
'''

ALLDEALS = r'''# ============== SCORECARD — re-price every saved deal ==============
import subprocess, pathlib
PY = r"C:\Users\CharlieJones\AppData\Local\Programs\Python\Python312\python.exe"
folder = pathlib.Path(r"C:\Users\CharlieJones\Downloads\Codes")
for w in ["BAML Write-Up.txt", "NYC_TFA_wire.txt", "psu_wire.txt",
          "portland_wire.txt", "ca_wire.txt"]:
    r = subprocess.run([PY, "run_pipeline.py", "--wire", w, "--no-archive"],
                       capture_output=True, text=True, cwd=folder)
    line = next((l for l in r.stdout.splitlines() if "mean abs" in l), "(failed)")
    print(f"{w:24s} {line.strip()}")
'''

TRACKER = r'''# ============== CONCESSION TRACKER ==============
import subprocess, pathlib
PY = r"C:\Users\CharlieJones\AppData\Local\Programs\Python\Python312\python.exe"
r = subprocess.run([PY, "concession_tracker.py"], capture_output=True, text=True,
                   cwd=r"C:\Users\CharlieJones\Downloads\Codes")
print(r.stdout or r.stderr[-2000:])
'''


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


nb = {"cells": [
    md("# Pricer\n"
       "**The daily driver.** Every cell launches a fresh process -- it can "
       "never be stale, never needs a kernel restart, and nothing here can "
       "overwrite the code.\n\n"
       "Three rules:\n"
       "1. New wire from Bloomberg -> paste it into `WIRE_TEXT` in the cell "
       "below, Shift+Enter.\n"
       "2. Saved deal -> put its filename in `WIRE`, keep `WIRE_TEXT` empty, "
       "Shift+Enter.\n"
       "3. Paste **dealer wires only** -- never a results table.\n"),
    code(RUNNER),
    md("## Extras"),
    code(ALLDEALS),
    code(TRACKER),
],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5}

out = HERE / "Pricer.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(nb['cells'])} cells)")
