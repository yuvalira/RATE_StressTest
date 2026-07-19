"""Two figures: the Experiment A replication of the paper's Fig 3, and the
Experiment B headline plot (estimator error vs. rewrite-direction
asymmetry, the novel stress test)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ESTIMATOR_LABELS = {"naive": "Naive", "single_rewrite": "Single-rewrite", "rate": "RATE"}
ESTIMATOR_COLORS = {"naive": "tab:blue", "single_rewrite": "tab:green", "rate": "tab:orange"}


def plot_experiment_a(df_long: pd.DataFrame, out_path: str, true_ate_value: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for est in ["naive", "single_rewrite", "rate"]:
        sub = df_long[df_long["estimator"] == est].sort_values("typo_level")
        ax.plot(sub["typo_level"], sub["mean_estimate"], marker="o", label=ESTIMATOR_LABELS[est], color=ESTIMATOR_COLORS[est])
    ax.axhline(true_ate_value, linestyle="--", color="black", linewidth=1, label="True ATE")
    ax.set_xlabel("Typo level (confound strength)")
    ax.set_ylabel("Estimated ATE")
    ax.set_title("Experiment A: replication of paper's Fig. 3")
    ax.legend()

    ax = axes[1]
    for est in ["naive", "single_rewrite", "rate"]:
        sub = df_long[df_long["estimator"] == est].sort_values("typo_level")
        ax.plot(sub["typo_level"], sub["rmse"], marker="o", label=ESTIMATOR_LABELS[est], color=ESTIMATOR_COLORS[est])
    ax.set_xlabel("Typo level (confound strength)")
    ax.set_ylabel("RMSE vs. true ATE")
    ax.set_title("Estimator error vs. confound strength")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_experiment_b(df_long: pd.DataFrame, out_path: str, metric: str = "rmse") -> None:
    df_long = df_long.copy()
    df_long["delta"] = df_long["p01"] - df_long["p10"]
    confound_levels = sorted(df_long["confound_strength"].unique())

    fig, axes = plt.subplots(1, len(confound_levels), figsize=(4.2 * len(confound_levels), 4.5), sharey=True)
    if len(confound_levels) == 1:
        axes = [axes]

    for ax, cs in zip(axes, confound_levels):
        panel = df_long[df_long["confound_strength"] == cs]
        for est in ["naive", "single_rewrite", "rate"]:
            sub = panel[panel["estimator"] == est].sort_values("delta")
            ax.plot(sub["delta"], sub[metric], marker="o", label=ESTIMATOR_LABELS[est], color=ESTIMATOR_COLORS[est])
        ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
        ax.set_xlabel(r"Asymmetry $\Delta = p_{01} - p_{10}$")
        ax.set_title(f"confound_strength = {cs}")

    axes[0].set_ylabel(metric.upper())
    axes[-1].legend()
    fig.suptitle("Experiment B: estimator error vs. rewrite-direction asymmetry")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
