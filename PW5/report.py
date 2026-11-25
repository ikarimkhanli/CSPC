import pandas as pd

ev = pd.read_csv("results/events.csv")

lines = []
for _, r in ev.iterrows():
    lines.append(f"Earthquake {r['event']}")
    lines.append(f"  P arrival at {r['tP_s']:.2f} s, S arrival at {r['tS_s']:.2f} s (S-P = {r['tS_s'] - r['tP_s']:.2f} s)")
    lines.append(f"  distance R = {r['R_km']:.1f} km, origin time = {r['origin_s']:.2f} s")
    lines.append(f"  PGV = {r['PGV_cm_s']:.3f} cm/s, magnitude M = {r['M']:.2f}")
    lines.append("")

closer = ev.loc[ev["R_km"].idxmin(), "event"]
stronger = ev.loc[ev["M"].idxmax(), "event"]
lines.append(f"Closer earthquake: {closer}")
lines.append(f"Stronger earthquake: {stronger}")

with open("results/report.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
