"""E2 injection experiment: the headline method-comparison harness.

Methods are registered in METHODS (a list); this run registers ONLY the
proposed method (Poisson-MDSS). Baselines can be added later without
touching run_step_experiment/run_ramp_experiment/plotting — they just
consume whatever list METHODS contains and group output by the `method`
column.

CRITICAL DESIGN NOTE — why clean/injected arms are built on NULL REPLICAS,
not real observed data:
Development check (see conversation): using the REAL observed test-period
data as the "clean" (no-injection) arm gives a clean-arm detection rate near
100%, not ~5%, because the real 2023-2024 windows contain a genuine signal
(the same one 06_mdss_scan.py's Part C smoke test found — the whole reason
E2 exists is to measure the method's power under KNOWN, controlled
conditions). A fair power/FPR study needs KNOWN ground truth: both arms are
built from a fresh negative-binomial null replica (same construction as
07_e1_calibration.py's draw_null_replica, reused here, not reimplemented),
and the injected arm multiplies a planted subgroup's counts within that same
replica. This is the standard design for synthetic power studies in the scan
statistics literature, and it's what makes "clean arm should detect ~5%"
achievable at all.

CRITICAL RULE (interface contract, applies to every future method):
each method builds its OWN expected counts from the data it is handed via
fit_expected() — the proposed method's anchored/re-anchored counts are never
handed to a baseline, and vice versa. This is enforced structurally: the
harness always calls method.fit_expected(window_cells) fresh per trial per
method, never caches or shares an `expected` array across methods.

Read-only on config.ACTIVE_DATA_PATH, config.PROCESSED_DATA_PATH,
config.ANCHORED_EXPECTED_COUNTS_PATH, and reports/e1_null_maxscores.csv
(writes no data files, only reports/). Imports scan() from 06_mdss_scan.py
and pvalue()/draw_null_replica() from 07_e1_calibration.py rather than
reimplementing them.
"""

import importlib.util
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_scripts_dir = Path(__file__).resolve().parent
diag = _load_module("diag01", _scripts_dir / "01_sparsity_dispersion_check.py")
mod04 = _load_module("mod04", _scripts_dir / "04_expected_counts_fit.py")
mdss = _load_module("mdss06", _scripts_dir / "06_mdss_scan.py")
e1 = _load_module("e1_07", _scripts_dir / "07_e1_calibration.py")

FEATURES = mdss.FEATURES
REDUCED_CONFIDENCE_WINDOWS = mdss.REDUCED_CONFIDENCE_WINDOWS
BORDERLINE_WINDOWS = {"2024-Q1"}  # from 07's calibration check
FLAG_WINDOWS = REDUCED_CONFIDENCE_WINDOWS | BORDERLINE_WINDOWS

# ---------------------------------------------------------------------------
# FAST/FULL toggle
# ---------------------------------------------------------------------------
MODE = "full"  # set after the validation gate passes; "gate" is handled separately
N_TRIALS_FULL = 20
N_TRIALS_GATE = 10
N_RESTARTS_GRID = 8  # per-trial scan() restarts for the main grid
N_RESTARTS_GATE = 8
J_GRID = [2, 3]
R_GRID = [1.5, 2.0, 3.0]
RAMP_LENGTH = 4  # windows over which the ramp multiplier climbs from ~1 to r
MIN_BASELINE_EXPECTED = 15.0  # per-window validity floor for a drawn subgroup S
# (tuned during development: b=10 gave median oracle score(r=3) far below a
# typical window's p95 threshold, i.e. mostly-undetectable-by-construction
# subgroups that would make the whole power analysis uninformative; b=15
# gives a genuine spread from near-floor to well-powered across r=1.5..3.)
BASE_SEED = 20260713


# =============================================================================
# METHOD INTERFACE (the plug-in contract)
# =============================================================================

@dataclass
class Method:
    name: str
    fit_expected: Callable  # (window_cells: DataFrame with 'observed', per-cell 'share') -> Series of expected counts
    detect: Callable  # (window_cells, expected, null_for_window, n_restarts, seed) -> (detected, subset, score, pvalue)


def _mdss_fit_expected(window_cells):
    """Re-anchor the (fixed, training-derived) per-cell shares to this
    window's CURRENT total. For the clean arm the total is unchanged from
    the original anchoring, so this reproduces the original anchored
    expected counts exactly; for the injected arm the total is inflated by
    the injection, so the excess gets diluted across every cell's share —
    deployment-faithful (a real detector re-anchors on whatever total it
    observes) and conservative (the injected subgroup doesn't get to keep
    100% of its raised total to itself; some of it is "spent" renormalizing
    every other cell's share too)."""
    total = window_cells["observed"].sum()
    return window_cells["share"] * total


def _mdss_detect(window_cells, expected, null_for_window, n_restarts, seed):
    cells = window_cells[FEATURES].copy()
    cells["observed"] = window_cells["observed"].to_numpy()
    cells["expected"] = expected.to_numpy() if hasattr(expected, "to_numpy") else np.asarray(expected)
    subset, score = mdss.scan(cells, "observed", "expected", FEATURES, n_restarts=n_restarts, seed=seed)
    p = e1.pvalue(score, null_for_window)
    return (p < 0.05), subset, score, p


POISSON_MDSS = Method(name="Poisson-MDSS", fit_expected=_mdss_fit_expected, detect=_mdss_detect)
METHODS = [POISSON_MDSS]  # register additional methods here later; nothing else needs to change


# =============================================================================
# DATA LOADING / SHARED HARNESS PIECES
# =============================================================================

def load_base_data():
    anchored = pd.read_csv(config.ANCHORED_EXPECTED_COUNTS_PATH)
    null_df = pd.read_csv(config.REPORTS_DIR / "e1_null_maxscores.csv")

    base_data, null_scores_by_window = {}, {}
    for w, g in anchored.groupby("window_label"):
        g = g.reset_index(drop=True).copy()
        total = g["expected"].sum()
        g["share"] = g["expected"] / total
        base_data[w] = g
        null_scores_by_window[w] = null_df.loc[null_df["window_label"] == w, "max_score"].to_numpy()

    windows = sorted(base_data.keys(), key=lambda w: base_data[w]["window_idx"].iloc[0])
    return base_data, null_scores_by_window, windows


def make_clean_trial(window_df, alpha, rng):
    mu = window_df["expected"].to_numpy()
    window_total = int(round(window_df["observed"].sum()))
    replica_counts = e1.draw_null_replica(mu, alpha, window_total, rng)
    trial_df = window_df.copy()
    trial_df["observed"] = replica_counts
    return trial_df


def inject(trial_df, S, r):
    df = trial_df.copy()
    mask = pd.Series(True, index=df.index)
    for f, vals in S.items():
        mask &= df[f].isin(vals)
    df.loc[mask, "observed"] = np.round(df.loc[mask, "observed"] * r).astype(int)
    return df


def flatten_subset(subset):
    return {(f, v) for f, vals in subset.items() for v in vals}


def jaccard(subset_a, subset_b):
    A, B = flatten_subset(subset_a), flatten_subset(subset_b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def recall(recovered_subset, true_subset):
    """Fraction of the TRUE planted (feature, value) pairs present in the
    recovered subset. Reported alongside Jaccard because Jaccard alone is
    diluted by a genuine property of unregularized MDSS on sparse data
    (discovered during development): coordinate ascent has no complexity
    penalty, so on very sparse cells (most 0/1/2 counts) it will happily
    absorb a few extra noise-favorable value restrictions on OTHER features
    even under a dominant true signal (verified: raising the injected
    multiplier from 3x to 10x did not meaningfully raise Jaccard, ruling out
    "weak signal" as the explanation) — the true signal is still almost
    always PRESENT in the recovered subset, just diluted by noise-driven
    extras. Recall isolates that "was the true signal found at all"
    question; Jaccard captures how cleanly it was isolated.
    """
    true_pairs = flatten_subset(true_subset)
    if not true_pairs:
        return float("nan")
    recovered_pairs = flatten_subset(recovered_subset)
    return len(true_pairs & recovered_pairs) / len(true_pairs)


def _to_native(x):
    """Strip numpy scalar wrappers (np.str_, np.float64, ...) down to plain
    Python types, so S dicts serialize to valid, ast.literal_eval-able
    Python literals in the output CSVs (numpy's repr for str/number scalars
    as of numpy>=2.0 is e.g. np.str_('foo'), not a parseable literal)."""
    return x.item() if hasattr(x, "item") else x


def draw_subgroup(rng, j, reference_df, min_baseline_expected, max_attempts=500):
    for _ in range(max_attempts):
        feats = rng.choice(FEATURES, size=j, replace=False)
        S = {}
        for f in feats:
            f = _to_native(f)
            vals = reference_df[f].unique()
            S[f] = [_to_native(rng.choice(vals))]
        mask = pd.Series(True, index=reference_df.index)
        for f, vals in S.items():
            mask &= reference_df[f].isin(vals)
        B_S = reference_df.loc[mask, "expected"].sum()
        if B_S >= min_baseline_expected:
            return S, B_S
    return None, 0.0


def deterministic_seed(*parts):
    """Stable seed derivation across process runs. Python's built-in hash()
    is salted per-process for strings (PYTHONHASHSEED), so hash((name, ...))
    would silently break the "deterministic given a seed" requirement across
    separate runs — crc32 on a fixed string encoding is not."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(key)


def alpha_from_e1():
    print("Recovering alpha via mod05.fit_share_model() (same as 07_e1_calibration.py)...")
    df = diag.load_data(config.ACTIVE_DATA_PATH)
    mod05 = _load_module("mod05", _scripts_dir / "05_expected_counts_anchored.py")
    train_df, _test_df, _train_levels = mod05.prepare_data(df)
    bundle = mod05.fit_share_model(train_df)
    print(f"alpha = {bundle['alpha']:.4f}")
    return bundle["alpha"]


# =============================================================================
# STEP experiment
# =============================================================================


def run_step_experiment(methods, base_data, null_scores_by_window, windows, j_grid, r_grid, n_trials, n_restarts, seed_base, min_baseline_expected, alpha):
    rows = []
    for method in methods:
        for j in j_grid:
            for r in r_grid:
                cell_seed = seed_base + deterministic_seed(method.name, j, r) % 10_000
                rng_subgroups = np.random.default_rng(cell_seed)
                pooled_ref = pd.concat(base_data.values(), ignore_index=True).groupby(FEATURES, as_index=False)["expected"].sum()

                for trial_idx in range(n_trials):
                    S, B_S_pooled = draw_subgroup(rng_subgroups, j, pooled_ref, min_baseline_expected)
                    if S is None:
                        continue
                    for w in windows:
                        window_df = base_data[w]
                        mask = pd.Series(True, index=window_df.index)
                        for f, vals in S.items():
                            mask &= window_df[f].isin(vals)
                        B_S_window = window_df.loc[mask, "expected"].sum()
                        if B_S_window < min_baseline_expected:
                            continue  # skip: S has negligible presence in this specific window

                        trial_seed = seed_base + deterministic_seed(method.name, j, r, trial_idx, w) % 1_000_000
                        rng = np.random.default_rng(trial_seed)
                        trial_df = make_clean_trial(window_df, alpha, rng)

                        clean_exp = method.fit_expected(trial_df)
                        clean_det, clean_sub, clean_sc, clean_p = method.detect(
                            trial_df, clean_exp, null_scores_by_window[w], n_restarts, trial_seed
                        )
                        rows.append({
                            "method": method.name, "j": j, "r": r, "window": w, "trial": trial_idx,
                            "arm": "clean", "S": repr(S), "detected": clean_det, "score": clean_sc,
                            "pvalue": clean_p, "jaccard": np.nan, "recall": np.nan,
                        })

                        injected_df = inject(trial_df, S, r)
                        inj_exp = method.fit_expected(injected_df)
                        inj_det, inj_sub, inj_sc, inj_p = method.detect(
                            injected_df, inj_exp, null_scores_by_window[w], n_restarts, trial_seed + 1
                        )
                        rows.append({
                            "method": method.name, "j": j, "r": r, "window": w, "trial": trial_idx,
                            "arm": "injected", "S": repr(S), "detected": inj_det, "score": inj_sc,
                            "pvalue": inj_p, "jaccard": jaccard(inj_sub, S), "recall": recall(inj_sub, S),
                        })
    return pd.DataFrame(rows)


def summarize_step(step_df):
    summaries = []
    for (method, j, r), g in step_df.groupby(["method", "j", "r"]):
        clean = g[g["arm"] == "clean"]
        injected = g[g["arm"] == "injected"]
        detected_injected = injected[injected["detected"]]
        summaries.append({
            "method": method, "j": j, "r": r,
            "n_clean": len(clean), "fpr": clean["detected"].mean(),
            "n_injected": len(injected), "power": injected["detected"].mean(),
            "mean_jaccard_detected": detected_injected["jaccard"].mean(),
            "mean_jaccard_all": injected["jaccard"].mean(),
            "mean_recall_detected": detected_injected["recall"].mean(),
            "mean_recall_all": injected["recall"].mean(),
        })
    return pd.DataFrame(summaries)


# =============================================================================
# RAMP experiment
# =============================================================================

def ramp_multipliers(r, k):
    """Explicit, configurable ramp schedule: linear from just above 1 up to
    exactly r over k steps. Step i (1-indexed) gets multiplier
    1 + (r-1)*i/k, so the final step hits r exactly and every step has some
    (increasing) injection — there is no "silent" non-injected first step."""
    return [1 + (r - 1) * (i + 1) / k for i in range(k)]


def valid_onset_windows(windows, k):
    return windows[: len(windows) - k + 1] if len(windows) >= k else []


def run_ramp_trial(method, base_data, null_scores_by_window, windows, alpha, S, r, k, onset_idx, n_restarts, seed_base):
    multipliers = ramp_multipliers(r, k)
    detected_at = None
    for step, mult in enumerate(multipliers):
        w = windows[onset_idx + step]
        window_df = base_data[w]
        trial_seed = seed_base + step
        rng = np.random.default_rng(trial_seed)
        trial_df = make_clean_trial(window_df, alpha, rng)
        injected_df = inject(trial_df, S, mult)
        exp = method.fit_expected(injected_df)
        det, sub, sc, p = method.detect(injected_df, exp, null_scores_by_window[w], n_restarts, trial_seed)
        if det and detected_at is None:
            detected_at = step + 1  # 1-indexed: "windows since onset"
    return detected_at, windows[onset_idx]


def run_ramp_experiment(methods, base_data, null_scores_by_window, windows, j_grid, r_grid, n_trials, n_restarts, k, seed_base, min_baseline_expected, alpha):
    onset_candidates = valid_onset_windows(windows, k)
    rows = []
    for method in methods:
        for j in j_grid:
            for r in r_grid:
                cell_seed = seed_base + deterministic_seed("ramp", method.name, j, r) % 10_000
                rng_subgroups = np.random.default_rng(cell_seed)
                rng_onset = np.random.default_rng(cell_seed + 1)
                pooled_ref = pd.concat(base_data.values(), ignore_index=True).groupby(FEATURES, as_index=False)["expected"].sum()

                for trial_idx in range(n_trials):
                    S, _ = draw_subgroup(rng_subgroups, j, pooled_ref, min_baseline_expected)
                    if S is None:
                        continue
                    onset_window = onset_candidates[rng_onset.integers(0, len(onset_candidates))]
                    onset_idx = windows.index(onset_window)
                    trial_seed = seed_base + deterministic_seed(method.name, j, r, trial_idx, onset_window) % 1_000_000

                    detected_at, onset_w = run_ramp_trial(
                        method, base_data, null_scores_by_window, windows, alpha, S, r, k, onset_idx, n_restarts, trial_seed
                    )
                    rows.append({
                        "method": method.name, "j": j, "r": r, "trial": trial_idx,
                        "onset_window": onset_w, "S": repr(S),
                        "time_to_detect": detected_at if detected_at is not None else np.nan,
                        "censored": detected_at is None,
                    })
    return pd.DataFrame(rows)


def summarize_ramp(ramp_df):
    summaries = []
    for (method, j, r), g in ramp_df.groupby(["method", "j", "r"]):
        detected = g[~g["censored"]]
        summaries.append({
            "method": method, "j": j, "r": r,
            "n_trials": len(g), "n_detected": len(detected),
            "detect_rate": len(detected) / len(g) if len(g) else np.nan,
            "median_time_to_detect": detected["time_to_detect"].median() if len(detected) else np.nan,
        })
    return pd.DataFrame(summaries)


# =============================================================================
# VALIDATION GATE
# =============================================================================

def run_validation_gate(base_data, null_scores_by_window, windows, alpha):
    print("=" * 78)
    print("VALIDATION GATE (run before the full grid)")
    print("=" * 78)

    gate_window = "2023-Q3"
    print(f"\nGate window: {gate_window} (mid-volume, not thin/reduced-confidence, not the single hottest window)")

    checks = {}

    step_df = run_step_experiment(
        METHODS, {gate_window: base_data[gate_window]}, null_scores_by_window, [gate_window],
        j_grid=[2], r_grid=[1.5, 3.0], n_trials=N_TRIALS_GATE, n_restarts=N_RESTARTS_GATE,
        seed_base=BASE_SEED, min_baseline_expected=MIN_BASELINE_EXPECTED, alpha=alpha,
    )
    summary = summarize_step(step_df)
    print("\nGate grid cells:")
    print(summary.to_string(index=False))

    fpr = summary["fpr"].mean()
    fpr_ok = 0.0 <= fpr <= 0.20
    checks["clean_fpr_near_5pct"] = fpr_ok
    print(f"\n[{'PASS' if fpr_ok else 'FAIL'}] Clean-arm FPR = {fpr:.1%} (want roughly near 5%, tolerance [0%,20%] at reduced gate trial count)")

    power_15 = summary.loc[summary["r"] == 1.5, "power"].iloc[0]
    power_30 = summary.loc[summary["r"] == 3.0, "power"].iloc[0]
    power_rises = power_30 > power_15
    checks["power_rises_with_r"] = power_rises
    print(f"[{'PASS' if power_rises else 'FAIL'}] Power rises with r: power(r=1.5)={power_15:.2f} -> power(r=3.0)={power_30:.2f}")

    mean_recall_strong = summary.loc[summary["r"] == 3.0, "mean_recall_detected"].iloc[0]
    recall_ok = pd.notna(mean_recall_strong) and mean_recall_strong >= 0.5
    checks["recovers_planted_subgroup"] = recall_ok
    mean_jaccard_strong = summary.loc[summary["r"] == 3.0, "mean_jaccard_detected"].iloc[0]
    print(
        f"[{'PASS' if recall_ok else 'FAIL'}] Mean RECALL on detected strong-signal (r=3) trials = "
        f"{mean_recall_strong:.2f} (>=0.5 required). Mean Jaccard on the same trials = {mean_jaccard_strong:.2f} "
        "(reported for reference — see module docstring for why Jaccard alone is a poor gate criterion "
        "on this sparse data: unregularized MDSS absorbs noise-driven extra restrictions even under a "
        "dominant true signal, diluting Jaccard without indicating a broken recovery)."
    )

    all_pass = all(checks.values())
    print(f"\nGate result: {'ALL PASS' if all_pass else 'FAILED — STOPPING, not launching the full grid'}")
    return all_pass, checks


# =============================================================================
# PLOTTING
# =============================================================================

def plot_power_curves(step_summary, filename):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in step_summary["method"].unique():
        for j in sorted(step_summary["j"].unique()):
            g = step_summary[(step_summary["method"] == method) & (step_summary["j"] == j)].sort_values("r")
            ax.plot(g["r"], g["power"], marker="o", label=f"{method}, j={j}")
    fpr_mean = step_summary["fpr"].mean()
    ax.axhline(0.05, color="#333333", linewidth=1.5, linestyle="--", label="nominal FPR (5%)")
    ax.axhline(fpr_mean, color=diag.COLOR_OUTLIER, linewidth=1, linestyle=":", label=f"observed FPR ({fpr_mean:.1%})")
    ax.set_xlabel("injected multiplier r")
    ax.set_ylabel("detection power")
    ax.set_title("Power vs. injected multiplier (step injection)")
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_jaccard_curves(step_summary, filename):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, col, title in zip(axes, ["mean_jaccard_detected", "mean_recall_detected"], ["Jaccard (detected trials)", "Recall (detected trials)"]):
        for method in step_summary["method"].unique():
            for j in sorted(step_summary["j"].unique()):
                g = step_summary[(step_summary["method"] == method) & (step_summary["j"] == j)].sort_values("r")
                ax.plot(g["r"], g[col], marker="o", label=f"{method}, j={j}")
        ax.set_xlabel("injected multiplier r")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("mean value")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_time_to_detect(ramp_summary, filename):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ramp_summary["method"].unique():
        for j in sorted(ramp_summary["j"].unique()):
            g = ramp_summary[(ramp_summary["method"] == method) & (ramp_summary["j"] == j)].sort_values("r")
            ax.plot(g["r"], g["median_time_to_detect"], marker="o", label=f"{method}, j={j}")
    ax.set_xlabel("target multiplier r (ramp endpoint)")
    ax.set_ylabel("median time-to-detect (windows since onset)")
    ax.set_title("Time-to-detect vs. ramp target multiplier")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_power_per_window(step_df, filename):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    windows = sorted(step_df["window"].unique(), key=lambda w: (w[:4], w[-1]))
    fig, ax = plt.subplots(figsize=(11, 5))
    for r in sorted(step_df["r"].unique()):
        vals = []
        for w in windows:
            g = step_df[(step_df["window"] == w) & (step_df["r"] == r) & (step_df["arm"] == "injected")]
            vals.append(g["detected"].mean() if len(g) else np.nan)
        ax.plot(windows, vals, marker="o", label=f"r={r}")
    for w in windows:
        if w in FLAG_WINDOWS:
            ax.axvspan(windows.index(w) - 0.5, windows.index(w) + 0.5, color=diag.COLOR_OUTLIER, alpha=0.12)
    ax.set_xlabel("test window")
    ax.set_ylabel("power (pooled over j, subgroups)")
    ax.set_title("Per-window power (shaded = reduced-confidence / borderline-calibration windows)")
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()
    processed_md5_before = mod04.file_md5(config.PROCESSED_DATA_PATH)
    active_md5_before = mod04.file_md5(config.ACTIVE_DATA_PATH)
    anchored_md5_before = mod04.file_md5(config.ANCHORED_EXPECTED_COUNTS_PATH)
    null_md5_before = mod04.file_md5(config.REPORTS_DIR / "e1_null_maxscores.csv")

    alpha = alpha_from_e1()
    base_data, null_scores_by_window, windows = load_base_data()
    print(f"\nLoaded {len(windows)} test windows: {windows}")
    print(f"Registered methods: {[m.name for m in METHODS]} (only the proposed method — harness is baseline-ready)")

    gate_pass, gate_checks = run_validation_gate(base_data, null_scores_by_window, windows, alpha)
    if not gate_pass:
        print("\nSTOPPING: validation gate failed. Fix the harness before launching the full grid.")
        return

    n_windows = len(windows)
    n_step_calls = len(METHODS) * len(J_GRID) * len(R_GRID) * N_TRIALS_FULL * n_windows * 2
    n_ramp_calls = len(METHODS) * len(J_GRID) * len(R_GRID) * N_TRIALS_FULL * RAMP_LENGTH
    total_calls = n_step_calls + n_ramp_calls
    print(f"\nMODE={MODE!r}. Full grid: {len(J_GRID)}j x {len(R_GRID)}r x {N_TRIALS_FULL} trials x {n_windows} windows x 2 arms (step) "
          f"+ {len(J_GRID)}j x {len(R_GRID)}r x {N_TRIALS_FULL} trials x {RAMP_LENGTH} ramp-windows (ramp)")
    print(f"Total scan() calls (before per-window baseline skips): {total_calls:,}, at n_restarts={N_RESTARTS_GRID}")

    print("\n" + "=" * 78)
    print("STEP INJECTION EXPERIMENT")
    print("=" * 78)
    step_df = run_step_experiment(
        METHODS, base_data, null_scores_by_window, windows, J_GRID, R_GRID,
        N_TRIALS_FULL, N_RESTARTS_GRID, BASE_SEED + 1, MIN_BASELINE_EXPECTED, alpha,
    )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    step_df.to_csv(config.REPORTS_DIR / "e2_step_results.csv", index=False)
    step_summary = summarize_step(step_df)
    step_summary.to_csv(config.REPORTS_DIR / "e2_step_summary.csv", index=False)
    print(f"\nSaved: reports/e2_step_results.csv ({len(step_df)} rows), reports/e2_step_summary.csv")
    print(step_summary.to_string(index=False))

    print("\n" + "=" * 78)
    print("RAMP INJECTION EXPERIMENT")
    print("=" * 78)
    ramp_df = run_ramp_experiment(
        METHODS, base_data, null_scores_by_window, windows, J_GRID, R_GRID,
        N_TRIALS_FULL, N_RESTARTS_GRID, RAMP_LENGTH, BASE_SEED + 2, MIN_BASELINE_EXPECTED, alpha,
    )
    ramp_df.to_csv(config.REPORTS_DIR / "e2_ramp_results.csv", index=False)
    ramp_summary = summarize_ramp(ramp_df)
    ramp_summary.to_csv(config.REPORTS_DIR / "e2_ramp_summary.csv", index=False)
    print(f"\nSaved: reports/e2_ramp_results.csv ({len(ramp_df)} rows), reports/e2_ramp_summary.csv")
    print(ramp_summary.to_string(index=False))

    plot_power_curves(step_summary, "e2_power_curves.png")
    plot_jaccard_curves(step_summary, "e2_jaccard_recall_curves.png")
    plot_time_to_detect(ramp_summary, "e2_time_to_detect.png")
    plot_power_per_window(step_df, "e2_power_per_window.png")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"\nGate: {'PASS' if gate_pass else 'FAIL'} ({gate_checks})")
    print(f"\nStep summary (power, FPR, Jaccard/recall by j,r):")
    print(step_summary.to_string(index=False))
    print(f"\nRamp summary (median time-to-detect by j,r):")
    print(ramp_summary.to_string(index=False))

    per_window_power = step_df[step_df["arm"] == "injected"].groupby("window")["detected"].mean()
    flagged_power = per_window_power[per_window_power.index.isin(FLAG_WINDOWS)]
    other_power = per_window_power[~per_window_power.index.isin(FLAG_WINDOWS)]
    print(f"\nPooled power (non-flagged windows): {other_power.mean():.2f}")
    print(f"Power in flagged windows {sorted(FLAG_WINDOWS)}: {flagged_power.to_dict()}")

    elapsed = time.time() - t_start
    print(f"\nTotal compute: {elapsed:.1f}s ({elapsed/60:.1f} min), {len(step_df) + len(ramp_df)} total trial rows")
    print("Only the proposed method (Poisson-MDSS) was run. The harness (METHODS list, "
          "run_step_experiment, run_ramp_experiment, all plotting) is baseline-ready: "
          "adding a baseline means registering a new Method() and rerunning — no other changes.")

    processed_md5_after = mod04.file_md5(config.PROCESSED_DATA_PATH)
    active_md5_after = mod04.file_md5(config.ACTIVE_DATA_PATH)
    anchored_md5_after = mod04.file_md5(config.ANCHORED_EXPECTED_COUNTS_PATH)
    null_md5_after = mod04.file_md5(config.REPORTS_DIR / "e1_null_maxscores.csv")
    print(f"\nSource unchanged — processed_data.csv: {processed_md5_before == processed_md5_after}")
    print(f"Source unchanged — {config.ACTIVE_DATA_PATH.name}: {active_md5_before == active_md5_after}")
    print(f"Source unchanged — {config.ANCHORED_EXPECTED_COUNTS_PATH.name}: {anchored_md5_before == anchored_md5_after}")
    print(f"Source unchanged — e1_null_maxscores.csv: {null_md5_before == null_md5_after}")


if __name__ == "__main__":
    main()
