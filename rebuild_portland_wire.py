"""Rebuild the Portland Water wire as its own permanent file from its
archived pricing run (real data, reformatted)."""
import glob
import pandas as pd

f = sorted(glob.glob('wire_archive/CITY_OF_PORTLAND*.csv'))[-1]
d = pd.read_csv(f)
lines = ['RE: $ 585,630,000*', 'CITY OF PORTLAND, OREGON',
         'SECOND LIEN WATER SYSTEM', 'REVENUE AND REFUNDING BONDS',
         '2026 SERIES A', '(GREEN BONDS)', '',
         "MOODY'S: Aa2 (Stable)                   S&P:   AA+ (Stable)",
         'FITCH:   NR                             KROLL: NR',
         'DATED:09/09/2026', 'DUE: 04/01', '']
for _, r in d.iterrows():
    amt = f"{int(r['Amount ($)'] / 1000):,}M"
    lines.append(f"{r['Maturity']}     {amt:>10}     {r['Coupon']:.2f}%     {r['Wire Yield']:.2f}")
    ptc = '' if r['Priced To'] == 'Maturity' else f"PTC {r['Priced To']} "
    lines.append(f"                      (Approx. $ Price {ptc}{r['Wire Price']:.3f})")
lines += ['', 'CALL FEATURES:  Optional call in 04/01/2036 @ 100.00']
open('portland_wire.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('portland_wire.txt rebuilt,', len(d), 'tranches')
