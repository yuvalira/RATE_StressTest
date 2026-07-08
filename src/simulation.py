"""Data generators for the two experiments.

Both produce a tidy DataFrame with the schema the estimators expect:
    unit_id, W, R_original, R_rewrite, R_rewrite_of_rewrite
where "rewrite" means "rewritten to flip W" and "rewrite_of_rewrite"
means "rewritten a second time, back to the original W".

Experiment A replicates the paper's own confounding-strength test using
the already-generated IMDB CSVs (typos correlated with W, corrected
symmetrically regardless of rewrite direction -- Assumption 1 holds by
construction, matching the paper's real GPT-4o rewriter).

Experiment B is the novel stress test: a fully synthetic simulation (no
text) with two orthogonal knobs -- confound_strength (correlation between
an off-target flag and W) and asymmetry Delta = p01 - p10 (how much the
rewriter's off-target error rate depends on rewrite direction, i.e. a
direct violation of Assumption 1).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from reward import DEFAULT_BETA_TYPO, DEFAULT_BETA_W, DEFAULT_NOISE_SD, reward

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "Synthetic_Data" / "datasets_by_typo_level"


def _word_diff_fraction(original: pd.Series, modified: pd.Series) -> np.ndarray:
    """Fraction of words that differ position-by-position between two
    equal-length-word texts (the notebook's typo injection only swaps
    adjacent characters within a word, so word counts are preserved)."""
    out = np.zeros(len(original), dtype=float)
    for i, (a, b) in enumerate(zip(original, modified)):
        wa, wb = a.split(), b.split()
        if not wa:
            continue
        n_diff = sum(1 for x, y in zip(wa, wb) if x != y)
        out[i] = n_diff / len(wa)
    return out


def load_experiment_a(
    typo_level: float,
    data_dir: Path = DEFAULT_DATA_DIR,
    seed: int = 0,
    beta_w: float = DEFAULT_BETA_W,
    beta_typo: float = DEFAULT_BETA_TYPO,
    noise_sd: float = DEFAULT_NOISE_SD,
) -> pd.DataFrame:
    """Load one imdb_vowel_typos_level_{typo_level:.2f}.csv and build the
    (original, rewrite, rewrite_of_rewrite) reward triple per unit.

    `synthetic_text` in this CSV is the paper's Table 2 confounded
    *original* observation (typos baked into the base data, correlated
    with W=starts_with_vowel) -- it is not an LLM rewrite. We therefore
    map it to R_original, with has_typo_original = (synthetic_text !=
    original_text): True for a typo_level-dependent fraction of W=1 rows,
    always False for W=0 rows (typos are only ever injected into W=1).

    The actual rewrite / rewrite-of-rewrite steps are not present in the
    CSV (no LLM calls were made), so we synthesize them as the paper's
    real rewriter behaves: GPT-4o "always fixes typos no matter what",
    regardless of direction, so both has_typo_rewrite and
    has_typo_rewrite2 are always False. This makes Assumption 1 hold by
    construction here, matching the paper's real experiment and giving a
    valid sanity-check baseline against Fig 3: Naive/Single-rewrite pick
    up the confound through R_original (increasingly biased as typo_level
    grows), while RATE never references R_original and stays unbiased.
    """
    path = Path(data_dir) / f"imdb_vowel_typos_level_{typo_level:.2f}.csv"
    raw = pd.read_csv(path)

    rng = np.random.default_rng(seed)
    n = len(raw)

    w = raw["W_starts_with_vowel"].to_numpy(dtype=int)
    # Continuous typo severity (fraction of words changed) rather than a
    # binary flag, so bias grows smoothly with typo_level instead of
    # saturating as soon as a review has at least one typo'd word.
    has_typo_original = _word_diff_fraction(raw["original_text"], raw["synthetic_text"])
    has_typo_rewrite = np.zeros(n, dtype=float)
    has_typo_rewrite2 = np.zeros(n, dtype=float)

    r_original = reward(w, has_typo_original, rng, beta_w, beta_typo, noise_sd)
    r_rewrite = reward(1 - w, has_typo_rewrite, rng, beta_w, beta_typo, noise_sd)
    r_rewrite_of_rewrite = reward(w, has_typo_rewrite2, rng, beta_w, beta_typo, noise_sd)

    return pd.DataFrame(
        {
            "unit_id": raw["unit_id"].to_numpy(),
            "W": w,
            "has_typo_original": has_typo_original,
            "has_typo_rewrite": has_typo_rewrite,
            "has_typo_rewrite2": has_typo_rewrite2,
            "R_original": r_original,
            "R_rewrite": r_rewrite,
            "R_rewrite_of_rewrite": r_rewrite_of_rewrite,
        }
    )


def load_all_experiment_a(
    typo_levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40),
    n_seeds: int = 200,
    data_dir: Path = DEFAULT_DATA_DIR,
    **reward_kwargs,
) -> pd.DataFrame:
    """Loop typo_levels x seeds; reward noise is resampled per seed, the
    has_typo flags are fixed by the CSV/typo_level. Returns a long df with
    `typo_level` and `seed` columns appended."""
    frames = []
    for level in typo_levels:
        for seed in range(n_seeds):
            df = load_experiment_a(level, data_dir=data_dir, seed=seed, **reward_kwargs)
            df["typo_level"] = level
            df["seed"] = seed
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def simulate_experiment_b(
    n: int,
    confound_strength: float,
    p01: float,
    p10: float,
    seed: int,
    base_rate: float = 0.1,
    confound_scale: float = 0.8,
    beta_w: float = DEFAULT_BETA_W,
    beta_typo: float = DEFAULT_BETA_TYPO,
    noise_sd: float = DEFAULT_NOISE_SD,
) -> pd.DataFrame:
    """Pure boolean-flag simulation of the RATE pipeline, no text.

    p01 = P(rewrite carries a typo | rewriting 0 -> 1)
    p10 = P(rewrite carries a typo | rewriting 1 -> 0)
    Delta = p01 - p10 is the asymmetry lever that stress-tests Assumption 1.
    confound_strength controls how strongly has_typo_original correlates
    with W (the paper's own knob, orthogonal to Delta).
    """
    rng = np.random.default_rng(seed)

    w = rng.integers(0, 2, size=n)

    p_typo_orig = np.clip(base_rate + confound_strength * (w - 0.5) * confound_scale, 0.0, 1.0)
    has_typo_original = rng.random(n) < p_typo_orig

    # Rewrite step: direction-specific probability, independent of
    # has_typo_original (the rewriter's own error process determines it).
    p_rewrite = np.where(w == 0, p01, p10)
    has_typo_rewrite = rng.random(n) < p_rewrite

    # Return-trip step: uses the opposite direction's probability.
    p_rewrite2 = np.where(w == 0, p10, p01)
    has_typo_rewrite2 = rng.random(n) < p_rewrite2

    r_original = reward(w, has_typo_original, rng, beta_w, beta_typo, noise_sd)
    r_rewrite = reward(1 - w, has_typo_rewrite, rng, beta_w, beta_typo, noise_sd)
    r_rewrite_of_rewrite = reward(w, has_typo_rewrite2, rng, beta_w, beta_typo, noise_sd)

    return pd.DataFrame(
        {
            "unit_id": np.arange(n),
            "W": w,
            "has_typo_original": has_typo_original,
            "has_typo_rewrite": has_typo_rewrite,
            "has_typo_rewrite2": has_typo_rewrite2,
            "R_original": r_original,
            "R_rewrite": r_rewrite,
            "R_rewrite_of_rewrite": r_rewrite_of_rewrite,
            "confound_strength": confound_strength,
            "p01": p01,
            "p10": p10,
        }
    )
