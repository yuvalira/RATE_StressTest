"""Naive, single-rewrite, and RATE estimators of the ATT/ATU/ATE of an
attribute W on a reward R, following the RATE paper's Algorithm 1.

Every estimator here consumes a DataFrame with columns:
    W                    -- binary attribute, 0/1
    R_original           -- reward of the original response
    R_rewrite            -- reward of the response rewritten to flip W
    R_rewrite_of_rewrite -- reward of that rewrite, rewritten back to
                             the original W

and returns a dict {"ATT": ..., "ATU": ..., "ATE": ...}.

All three estimators combine ATT/ATU into ATE with the same weights,
(n1/n) for ATT and (n0/n) for ATU, per the law-of-total-expectation
identity ATE = P(W=1)*ATT + P(W=0)*ATU -- this identity does not depend
on which estimator produced ATT/ATU, so single_rewrite and rate use
identical weighting.
"""

import pandas as pd


def _group_sizes(df: pd.DataFrame) -> tuple[int, int, int]:
    n1 = int((df["W"] == 1).sum())
    n0 = int((df["W"] == 0).sum())
    return n1, n0, n1 + n0


def naive(df: pd.DataFrame) -> dict:
    """Correlational difference in means using only original responses."""
    r1 = df.loc[df["W"] == 1, "R_original"].mean()
    r0 = df.loc[df["W"] == 0, "R_original"].mean()
    ate = r1 - r0
    return {"ATT": ate, "ATU": ate, "ATE": ate}


def single_rewrite(df: pd.DataFrame) -> dict:
    """Estimator using (original, rewrite) pairs."""
    g1 = df[df["W"] == 1]
    g0 = df[df["W"] == 0]
    n1, n0, n = _group_sizes(df)

    att = (g1["R_original"] - g1["R_rewrite"]).mean()
    atu = (g0["R_rewrite"] - g0["R_original"]).mean()
    ate = (n1 / n) * att + (n0 / n) * atu
    return {"ATT": att, "ATU": atu, "ATE": ate}


def rate(df: pd.DataFrame) -> dict:
    """RATE estimator using (rewrite, rewrite-of-rewrite) pairs."""
    g1 = df[df["W"] == 1]
    g0 = df[df["W"] == 0]
    n1, n0, n = _group_sizes(df)

    att = (g1["R_rewrite_of_rewrite"] - g1["R_rewrite"]).mean()
    atu = (g0["R_rewrite"] - g0["R_rewrite_of_rewrite"]).mean()
    ate = (n1 / n) * att + (n0 / n) * atu
    return {"ATT": att, "ATU": atu, "ATE": ate}


ESTIMATORS = {"naive": naive, "single_rewrite": single_rewrite, "rate": rate}
