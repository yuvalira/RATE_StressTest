# RATE_StressTest

# RATE Stress Test

> **Empirical Evaluation of the Assumptions Behind RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals**

[![Course Project](https://img.shields.io/badge/Course-Causal%20Inference-blue)]()
[![Paper](https://img.shields.io/badge/Paper-RATE%20\(ICML%202025\)-green)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)]()

---

## Overview

This repository contains the empirical study for our final project in **Causal Inference in the AI Era**.

We investigate the robustness of **RATE (Rewrite-based Attribute Treatment Estimators)**, a causal framework proposed for understanding what reward models actually reward.

Rather than reproducing the paper's main experiments, we focus on a key theoretical assumption and examine when the method begins to fail.

---

## Paper

**RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals**

**Authors:** David Reber, Sean M. Richardson, Todd Nief, Cristina Garbacea, Victor Veitch

**Venue:** ICML 2025

---

## Research Question

RATE relies on a double-rewrite procedure:

```text
Original → Rewrite → Rewrite of Rewrite
```

The theory assumes that off-target rewrite errors are approximately symmetric across rewrite directions.

Our central question is:

> How robust is RATE when rewrite errors become increasingly asymmetric?

---

## Motivation

The original paper shows that RATE can successfully recover causal effects under confounding.

However, its identification guarantees depend on assumptions that are difficult to verify in real-world language models.

This project focuses on:

### Assumption 1 — Symmetric Rewrite Errors

Off-target changes introduced during rewriting should not depend on rewrite direction.

### Assumption 2 — Additive Reward Components

The effect of the target attribute should be separable from the effect of rewrite-induced changes.

Our primary stress test targets **Assumption 1**.

---

## Experimental Design

We create a controlled synthetic environment where:

✅ The true causal effect is known

✅ Confounding strength can be controlled

✅ Rewrite quality can be manipulated

✅ Multiple random seeds can be evaluated

For each setting we compare:

| Estimator      | Description                |
| -------------- | -------------------------- |
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
* The double-rewrite correction becomes less effective.

---

## Repository Structure

```text
RATE_StressTest/
│
├── notebooks/
│   └── rate_stress_test.ipynb
│
├── src/
│   ├── simulation.py
│   ├── estimators.py
│   ├── metrics.py
│   └── plotting.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── README.md
└── requirements.txt
```

---

## Evaluation Metrics

For every experiment we report:

* Average Treatment Effect (ATE)
* Bias
* Variance
* Mean Squared Error (MSE)

Results are averaged across multiple random seeds to ensure robustness.

---

## Expected Contribution

This study provides a focused empirical investigation of one of RATE's central assumptions.

Our goal is to identify the conditions under which the double-rewrite cancellation mechanism succeeds—and where it begins to break down.

---

## Authors

**Yuval Ratzabi**
**Tuvia Hausdorff**

Final Project — *Causal Inference in the AI Era*

