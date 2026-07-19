# RATE_StressTest

> **Empirical Evaluation of the Assumptions Behind RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals**

[![Course Project](https://img.shields.io/badge/Course-Causal%20Inference-blue)]()
[![Paper](https://img.shields.io/badge/Paper-RATE%20\(ICML%202025\)-green)]()

---

## Overview

This repository contains the empirical study for our final project in **Causal Inference in the AI Era**.

We investigate the robustness of **RATE (Rewrite-based Attribute Treatment Estimators)**, a causal framework proposed for understanding what reward models actually reward.

Rather than reproducing the paper's main experiments, we focus on a key theoretical assumption and examine when the method begins to fail.

---

## Paper

**RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals**

**Authors:** David Reber, Sean M. Richardson, Todd Nief, Cristina Garbacea, Victor Veitch

**Venue:** ICML 2025 (PMLR 267)

**Paper:** https://arxiv.org/abs/2410.13844

**Code:** https://github.com/toddnief/RATE

---

## Research Question

RATE relies on a double-rewrite procedure:

```text
Original → Rewrite → Rewrite of Rewrite
```

The method's validity theorem (Theorem 4.1 in the paper) rests on an assumption that off-target rewrite errors are drawn from a distribution that does not depend on rewrite direction (Assumption 1).

Our central question is:

> How does RATE behave when this direction-symmetry assumption is violated, and is this violation distinguishable from the confounding problem RATE was designed to solve?

---

## Motivation

The original paper shows that RATE can successfully recover causal effects under confounding between an off-target attribute (typos) and the attribute of interest, as long as the LLM rewriter corrects the off-target attribute the same way regardless of which direction it is rewriting. Their own experiments never vary that direction-symmetry itself — it holds by construction, because their real rewriter (GPT-4o) happens to correct typos "no matter what."

This project asks what happens when that assumption is dropped, using two experiments with known ground truth:

### Assumption 1 — Symmetric Rewrite Errors

Off-target changes introduced during rewriting should not depend on rewrite direction.

### Assumption 2 — Additive Reward Components

The effect of the target attribute should be separable from the effect of rewrite-induced changes.

Our experiments target **Assumption 1** directly; Assumption 2 is held to hold by construction (our synthetic reward is additive) so that any bias we observe can be attributed to Assumption 1 alone.

---

## Experimental Design

Two experiments, both with known ground-truth ATE so bias/variance/RMSE can be computed exactly:

**Experiment A — baseline replication.** Reuses the existing IMDB vowel/typo datasets (`Synthetic_Data/`) to reproduce the paper's own confounding-strength sweep (their Fig. 3). This is a sanity check: it confirms our estimator implementation reproduces known behavior (Naive and Single-rewrite biased and growing with confound strength, RATE flat near the true effect) before trusting it on anything new.

**Experiment B — the stress test.** A fully synthetic simulation (no text) with two independent knobs: confounding strength between an off-target flag and the attribute of interest (the paper's own knob), and an asymmetry parameter `Δ = p01 − p10` controlling how much the simulated rewriter's off-target error rate depends on rewrite direction (the untested knob). Both are swept independently, with many random seeds per grid point.

For each setting we compare:

| Estimator      | Description                |
| -------------- | --------------------------- |
| Naive          | Observational comparison   |
| Single Rewrite | One counterfactual rewrite |
| RATE           | Double-rewrite estimator   |

---

## Hypotheses

### H1 — Increasing Confounding

As confounding becomes stronger:

* Naive estimates become increasingly biased.
* RATE remains relatively stable.

### H2 — Breaking Symmetry

As rewrite errors become more direction-dependent:

* RATE's bias increases.
* The double-rewrite correction becomes less effective, independent of confounding strength.

---

## Repository Structure

```text
RATE_StressTest/
│
├── Synthetic_Data/
│   ├── RATE_Dataset_Generation.ipynb        # generates the IMDB vowel/typo datasets used by Experiment A
│   ├── datasets_by_typo_level/*.csv
│   └── synthetic_text_data_summary.csv
│
├── notebooks/
│   └── rate_stress_test.ipynb               # runs both experiments end to end
│
├── src/
│   ├── reward.py                            # synthetic reward function, known ground-truth ATE
│   ├── simulation.py                        # data generators for Experiments A and B
│   ├── estimators.py                        # naive / single-rewrite / RATE estimators
│   ├── metrics.py                           # bias / variance / RMSE, sweep runner
│   └── plotting.py                          # figures for both experiments
│
├── results/
│   ├── figures/
│   └── tables/
│
├── requirements.txt
├── requirements-report.txt
└── README.md
```

---

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name rate_stresstest --display-name "RATE StressTest"
```

## Running the experiments

Open `notebooks/rate_stress_test.ipynb` (select the `RATE StressTest` kernel) and run all cells. This will:

1. Hand-verify the three estimators against a toy example.
2. Run Experiment A and save `results/tables/experiment_a_results.csv` and `results/figures/experiment_a_replication.png`.
3. Run Experiment B and save `results/tables/experiment_b_results.csv` and `results/figures/experiment_b_headline.png`.

---

## Evaluation Metrics

For every experiment we report, across many random seeds per setting:

* Mean estimate and Bias against the known true ATE
* Variance
* Root Mean Squared Error (RMSE)

---

## Report

The written report is at `report/report.md`, rendered to `report/report.pdf`. To rebuild it after editing:

```bash
pip install -r requirements-report.txt
python report/build_diagram.py   # regenerates results/figures/potential_outcomes_diagram.png
python report/build_pdf.py       # renders report/report.md -> report/report.pdf
```

## Summary of Findings

Experiment A reproduces the paper's own result: Naive and Single-rewrite bias grow with confound strength while RATE stays flat near the true ATE.

Experiment B shows that RATE's error is not affected by confound strength at all, but grows roughly linearly in the asymmetry parameter Δ, in a way that matches a closed-form bias prediction derived from the synthetic reward model. This bias is present regardless of confounding strength, meaning direction-asymmetry is a failure mode distinct from the one the paper tests, and one that its own diagnostics (comparing marginal reward distributions before and after a round-trip rewrite) would not necessarily catch.

See `notebooks/rate_stress_test.ipynb` and `results/` for full detail.

---

## Authors

Yuval Ratzabi

Tuvia Hausdorff

Final Project — *Causal Inference in the AI Era*
