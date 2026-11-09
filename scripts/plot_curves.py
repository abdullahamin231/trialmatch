"""Plot training/validation curves from results/logs/train_*.csv -> figures/."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
logs = sorted((ROOT / "results" / "logs").glob("train_*.csv"))
if len(sys.argv) > 1:
    logs = [p for p in logs if sys.argv[1] in p.name]

latest = max(logs, key=lambda p: p.stat().st_mtime)
steps, loss, val_steps, val_loss = [], [], [], []
with latest.open() as fh:
    for row in csv.DictReader(fh):
        steps.append(int(row["step"]))
        loss.append(float(row["loss"]))
        if row.get("val_loss"):
            val_steps.append(int(row["step"]))
            val_loss.append(float(row["val_loss"]))

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(steps, loss, label="train loss", lw=1.5)
if val_steps:
    ax.plot(val_steps, val_loss, "o-", label="validation loss", lw=1.2, ms=3)
ax.set_xlabel("optimizer step")
ax.set_ylabel("loss (answer tokens only)")
ax.set_title(f"Fine-tuning curves ({latest.name})", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
out = ROOT / "results" / "figures" / "training_curve.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print("wrote", out)
