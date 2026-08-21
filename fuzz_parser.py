"""Adversarial parser probe: realistic wire oddities. The requirement is not
that every dialect parses -- it's that failures are LOUD (row skipped -> the
tranche-sum warning fires) and nothing crashes or silently misparses."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from Wire_Parser import parse_wire

CASES = {
    "yield with '+' (priced to par call)": """RE: $ 10,000,000
TEST
DATED:01/15/2027
06/01/2040     10,000M     5.00%     3.57+
                      (Approx. $ Price 100.000)
""",
    "sinking fund 'S' marker": """RE: $ 10,000,000
TEST
DATED:01/15/2027
06/01/2040 S   10,000M     5.00%     4.10      0.40
                      (Approx. $ Price 106.000)
""",
    "price on same line": """RE: $ 10,000,000
TEST
DATED:01/15/2027
06/01/2040     10,000M     5.00%     4.10      0.40 (Approx. $ Price 106.000)
""",
    "two-digit yield": """RE: $ 10,000,000
TEST
DATED:01/15/2027
06/01/2040     10,000M     6.00%     10.25      0.40
                      (Approx. $ Price 71.500)
""",
    "comma in takedown / extra cols": """RE: $ 10,000,000
TEST
DATED:01/15/2027
06/01/2040     10,000M     5.00%     4.10      0.40   AAA/AA+
                      (Approx. $ Price 106.000)
""",
    "empty wire": "",
    "no maturities at all": "RE: $ 10,000,000\nJUST A HEADER\nDATED:01/15/2027\n",
}

crashes = 0
for name, text in CASES.items():
    try:
        d = parse_wire(text)
        n = len(d['tranches'])
        priced = sum(1 for t in d['tranches'] if t['price'] is not None)
        total = sum(t['amount'] for t in d['tranches'])
        loud = (d['issue_amount'] or 0) != total
        print(f"[{'LOUD' if (n == 0 or loud or priced < n) else 'ok  '}] {name}: "
              f"{n} tranche(s), {priced} priced, sum-check {'FIRES' if loud else 'quiet'}")
    except Exception as e:
        crashes += 1
        print(f"[CRASH] {name}: {type(e).__name__}: {e}")

print(f"\n{crashes} crash(es). Requirement: zero crashes; unparsed rows must be LOUD.")
sys.exit(1 if crashes else 0)
