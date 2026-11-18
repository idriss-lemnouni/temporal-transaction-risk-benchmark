import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

PALETTE = ("#067a78", "#2554a4", "#a65316")
MODELS = {"logistic": "Régression logistique", "boosting": "Boosting", "graphsage": "GraphSAGE"}


def save(fig, output, name):
    fig.savefig(output / name, dpi=160, facecolor="#f7f9fc")
    plt.close(fig)


def quality(metrics, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    fig.suptitle("Qualité du classement et de la calibration", x=.02, ha="left",
                 fontsize=17, fontweight="bold")
    for ax, key, title in zip(axes, ("average_precision", "brier"),
                              ("Précision moyenne ↑", "Score de Brier ↓"), strict=True):
        values = [result["test"][key] for result in metrics["models"].values()]
        names = [MODELS[name] for name in metrics["models"]]
        bars = ax.barh(names, values, color=PALETTE, height=.58)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlim(0, max(values) * 1.27)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.invert_yaxis()
    fig.supxlabel("Test Elliptic · pas 35 à 49 · transactions annotées", fontsize=10)
    save(fig, output, "ranking-and-calibration.png")


def budget_summary(metrics, output):
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    budget = metrics["selected_budget"]
    ax.set_title(f"Composition des {budget} revues par pas de temps", loc="left",
                 fontsize=17, fontweight="bold")
    names = [MODELS[name] for name in metrics["models"]]
    reviews = [result["test"]["review"] for result in metrics["models"].values()]
    offset = np.zeros(len(reviews))
    for key, color, label in (("illicit", "#067a78", "Illicite confirmé"),
                              ("licit", "#c97235", "Licite confirmé"),
                              ("unknown", "#abb6cb", "Label inconnu")):
        values = [review[key] for review in reviews]
        bars = ax.barh(names, values, left=offset, color=color, label=label, height=.55)
        ax.bar_label(bars, labels=[str(value) if value > 0.025 * max(r["total"] for r in reviews)
                                  else "" for value in values], label_type="center",
                     color="#152139" if key != "illicit" else "white", fontsize=10)
        offset += values
    ax.set_xlabel("Transactions sélectionnées sur les 15 pas de test")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=3, frameon=False)
    ax.invert_yaxis()
    save(fig, output, "review-budget.png")


def drift(metrics, output):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    fig.suptitle("Le classement et la couverture des labels changent dans le temps",
                 x=.02, ha="left", fontsize=15, fontweight="bold")
    for (name, result), color in zip(metrics["models"].items(), PALETTE, strict=True):
        rows = result["test_steps"]
        steps = np.array([int(item) for item in rows])
        top.plot(steps, [row["average_precision"] for row in rows.values()],
                 marker="o", ms=4, color=color, label=MODELS[name])
    bottom.bar(steps, [row["annotation_fraction"] for row in rows.values()],
               width=.62, color="#637ca2", alpha=.9)
    top.set_ylabel("Précision moyenne")
    top.legend(loc="upper center", bbox_to_anchor=(.5, -.05), ncol=3, frameon=False)
    bottom.set_ylabel("Part annotée")
    bottom.yaxis.set_major_formatter(PercentFormatter(1))
    bottom.set_xlabel("Pas de temps du test")
    bottom.set_xticks(steps)
    save(fig, output, "temporal-drift.png")


def main():
    parser = argparse.ArgumentParser(description="Draw aggregate Elliptic benchmark figures")
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("figures/results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.axisbelow": True, "grid.alpha": .25,
                         "axes.facecolor": "#f7f9fc", "text.color": "#152139",
                         "axes.labelcolor": "#152139", "xtick.color": "#32455d",
                         "ytick.color": "#32455d", "figure.facecolor": "#f7f9fc"})
    quality(metrics, args.output)
    budget_summary(metrics, args.output)
    drift(metrics, args.output)


if __name__ == "__main__":
    main()
