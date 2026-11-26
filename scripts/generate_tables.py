import argparse, csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with a.input.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rr = [
        r
        for r in rows
        if r.get("contract") == "operational" and r.get("subset", "full") == "full"
    ]
    fields = ["task", "arm", "mean", "sd"]
    text = "\\begin{tabular}{llrr}\nTask & Model & Mean & SD \\\\\n"
    for r in rr:
        text += f"{r['task']} & {r['arm']} & {100*float(r['mean']):.4f} & {100*float(r['sd']):.4f} \\\\\n"
    text += "\\end{tabular}\n"
    (a.out / "main_table.tex").write_text(text, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(
        [r["task"] + " " + r["arm"] for r in rr], [100 * float(r["mean"]) for r in rr]
    )
    ax.set_ylabel("Macro-F1 (%)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(a.out / "main_results.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
