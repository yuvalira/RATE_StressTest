"""Builds results/figures/potential_outcomes_diagram.png: the potential-
outcomes setup and RATE's double-rewrite procedure, side by side."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "results" / "figures" / "potential_outcomes_diagram.png"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))


def box(ax, xy, w, h, text, fc="white", ec="black", fontsize=10.5):
    x, y = xy
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def arrow(ax, p_from, p_to, color="black", lw=1.3):
    a = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=14,
                         color=color, linewidth=lw, zorder=1)
    ax.add_patch(a)


# -------- Panel 1: potential-outcomes DAG --------
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Potential-outcomes setup", fontsize=12, fontweight="bold")

box(ax, (1.6, 8.3), 2.3, 1.1, "Prompt $X$")
box(ax, (1.6, 5.5), 2.3, 1.1, "Attribute $W$\n(e.g. sentiment)")
box(ax, (5.2, 6.9), 2.7, 1.3, "Response\n$Y(W)$", fc="#eaf2ff")
box(ax, (8.6, 6.9), 2.4, 1.3, "Reward\n$R(X,Y(W))$", fc="#fff3e0")

arrow(ax, (2.75, 8.0), (3.85, 7.3))
arrow(ax, (2.75, 5.8), (3.85, 6.6))
arrow(ax, (6.55, 6.9), (7.4, 6.9))

ax.text(2.0, 3.6,
        "Only one of $Y(1)$, $Y(0)$\nis ever observed per unit\n"
        "(fundamental problem of\ncausal inference)",
        ha="center", va="center", fontsize=8.6, style="italic", color="dimgray")

ax.text(6.9, 3.6,
        "$\\mathrm{ATT}=\\mathbb{E}[R(X,Y(1))-R(X,Y(0))\\mid W=1]$\n"
        "$\\mathrm{ATU}=\\mathbb{E}[R(X,Y(1))-R(X,Y(0))\\mid W=0]$",
        ha="center", va="center", fontsize=9.3)

ax.text(5.0, 1.3,
        r"Estimand:  $\mathrm{ATE}=\mathbb{E}[R(X,Y(1))-R(X,Y(0))] = P(W{=}1)\cdot\mathrm{ATT} + P(W{=}0)\cdot\mathrm{ATU}$",
        ha="center", va="center", fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="gray"))

# -------- Panel 2: RATE double-rewrite procedure --------
ax = axes[1]
ax.set_xlim(0, 11)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("RATE: double-rewrite procedure ($W=w$ for this unit)", fontsize=12, fontweight="bold")

box(ax, (1.5, 7.6), 2.6, 1.3, "Original $Y$\nattribute $=w$", fc="#eaf2ff")
box(ax, (5.5, 7.6), 2.6, 1.3, "Rewrite $\\widetilde{Y}$\nattribute $=1{-}w$", fc="#e8f8ec")
box(ax, (9.5, 7.6), 2.6, 1.3, "Rewrite of rewrite\nattribute $=w$", fc="#e8f8ec")

arrow(ax, (2.8, 7.6), (4.2, 7.6))
ax.text(3.5, 9.1, "rewrite", ha="center", fontsize=8.7)
ax.text(3.5, 8.7, "off-target error $\\tilde\\xi$", ha="center", fontsize=8, color="dimgray")

arrow(ax, (6.8, 7.6), (8.2, 7.6))
ax.text(7.5, 9.1, "rewrite back", ha="center", fontsize=8.7)
ax.text(7.5, 8.7, "off-target error $\\tilde\\xi'$", ha="center", fontsize=8, color="dimgray")

arrow(ax, (1.5, 6.9), (1.5, 6.2), color="gray")
arrow(ax, (5.5, 6.9), (5.5, 6.2), color="gray")
arrow(ax, (9.5, 6.9), (9.5, 6.2), color="gray")
ax.text(1.5, 5.7, "$R(Y)$", ha="center", fontsize=11)
ax.text(5.5, 5.7, "$R(\\widetilde{Y})$", ha="center", fontsize=11)
ax.text(9.5, 5.7, "$R(\\widetilde{\\widetilde{Y}})$", ha="center", fontsize=11)

ax.annotate("", xy=(1.65, 4.3), xytext=(5.35, 4.3),
            arrowprops=dict(arrowstyle="<->", color="firebrick", linewidth=1.3))
ax.text(3.5, 3.7, "Naive / single-rewrite\ncompare $R(Y)$ vs. $R(\\widetilde{Y})$",
        ha="center", fontsize=8.8, color="firebrick")

ax.annotate("", xy=(5.65, 2.3), xytext=(9.35, 2.3),
            arrowprops=dict(arrowstyle="<->", color="darkgreen", linewidth=1.3))
ax.text(7.5, 1.6, "RATE instead compares $R(\\widetilde{Y})$ vs. $R(\\widetilde{\\widetilde{Y}})$\n"
                  "valid only if the rewrite error is direction-symmetric (Assumption 1)",
        ha="center", fontsize=8.4, color="darkgreen")

fig.subplots_adjust(left=0.035, right=0.98, top=0.90, bottom=0.03, wspace=0.15)
fig.savefig(OUT, dpi=170)
print("wrote", OUT)
