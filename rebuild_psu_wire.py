"""Rebuild the Penn State wire text file from its archived pricing run
(real data, reformatted) so it can be re-priced under the tenor ramp."""
import glob
import pandas as pd

f = sorted(glob.glob('wire_archive/THE_PENNSYLVANIA*.csv'))[-1]
d = pd.read_csv(f)
lines = ['RE: $ 394,650,000*', 'THE PENNSYLVANIA STATE UNIVERSITY',
         'Bonds, Tax-Exempt Series A of 2026', '',
         "MOODY'S: Aa1 (Stable)                   S&P:   AA (Stable)",
         'FITCH:   NR                             KROLL: NR',
         'DATED:09/17/2026   FIRST COUPON:03/01/2027', 'DUE: 09/01', '']
for _, r in d.iterrows():
    amt = f"{int(r['Amount ($)'] / 1000):,}M"
    lines.append(f"{r['Maturity']}     {amt:>10}     {r['Coupon']:.2f}%     {r['Wire Yield']:.2f}")
    ptc = '' if r['Priced To'] == 'Maturity' else f"PTC {r['Priced To']} "
    lines.append(f"                      (Approx. $ Price {ptc}{r['Wire Price']:.3f})")
lines += ['', 'CALL FEATURES:  Optional call in 09/01/2036 @ 100.00']
open('psu_wire.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('psu_wire.txt rebuilt,', len(d), 'tranches')
