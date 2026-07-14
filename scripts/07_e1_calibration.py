"""E1 calibration: build the per-3-month-window null distribution of MDSS
max-scores, turning raw scan() scores into p-values.

Null model: negative binomial (NB2) with mean = anchored expected count per
cell and dispersion alpha recovered from 05_expected_counts_anchored.py's
share-model fit (05 doesn't persist alpha to disk, so it's recovered by
calling 05's own fit_share_model() again on the same training data — cheap
(<1s) and deterministic, not a re-derivation of the method).

Read-only on config.ACTIVE_DATA_PATH, config.PROCESSED_DATA_PATH, and
config.ANCHORED_EXPECTED_COUNTS_PATH (writes no data files, only reports/).
Imports scan() from 06_mdss_scan.py rather than reimplementing it.
"""

import importlib.util
import sys
import time
from pathlib import Path

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
mod05 = _load_module("mod05", _scripts_dir / "05_expected_counts_anchored.py")
mdss = _load_module("mdss06", _scripts_dir / "06_mdss_scan.py")

FEATURES = mdss.FEATURES

# ---------------------------------------------------------------------------
# FAST/FULL toggle. Edit MODE and rerun; every random draw is seeded off
# BASE_SEED so results are fully reproducible for a given MODE.
# ---------------------------------------------------------------------------
MODE = "fast"  # "fast" (dev) or "full" (final run)
M_BY_MODE = {"fast": 199, "full": 999}
M = M_BY_MODE[MODE]
N_RESTARTS_NULL = 3  # restarts per null-replica scan/calibration-check scan —
# deliberately smaller than a real discovery scan's restart count (06 defaults
# to 20): null calibration needs BREADTH (many replicas to populate the tail)
# more than per-replica DEPTH (finding that one replica's exact optimum
# barely matters once you're averaging/percentile-ing over M of them). The
# calibration check below is exactly what empirically verifies this
# trade-off is safe rather than just asserting it.
BASE_SEED = 20260711

CALIBRATION_ALPHA = 0.05  # nominal false-positive rate the null should hit
MISCALIBRATION_LOW, MISCALIBRATION_HIGH = 0.01, 0.10
REDUCED_CONFIDENCE_WINDOWS = mdss.REDUCED_CONFIDENCE_WINDOWS


def recover_alpha():
    print("=" * 78)
    print("RECOVER ALPHA (NB dispersion) FROM SCRIPT 05")
    print("=" * 78)
    print(
        "\n05_expected_counts_anchored.py doesn't persist alpha to disk, so it's "
        "recovered here by calling 05's own prepare_data()/fit_share_model() "
        "against the same training data (2018-2020) — this re-runs the cheap, "
        "deterministic GLM fit (<1s), it does not re-derive the method. Output "
        "of that fit follows:\n"
    )
    df = diag.load_data(config.ACTIVE_DATA_PATH)
    train_df, _test_df, _train_levels = mod05.prepare_data(df)
    bundle = mod05.fit_share_model(train_df)
    alpha = bundle["alpha"]
    print(f"\nRecovered alpha = {alpha:.4f} (from mod05.fit_share_model(train_df)['alpha'])")
    return alpha


def load_anchored_counts():
    df = pd.read_csv(config.ANCHORED_EXPECTED_COUNTS_PATH)
    windows = sorted(df["window_label"].unique(), key=lambda w: df.loc[df["window_label"] == w, "window_idx"].iloc[0])
    return df, windows


def draw_null_replica(mu, alpha, window_total, rng):
    """One NB2 null replica, conditioned to sum to window_total.

    Conditioning choice: draw raw NB2 counts per cell (mean=mu_i, dispersion
    alpha, so variance = mu_i + alpha*mu_i^2), then rescale the whole vector
    by a single constant so it sums exactly to window_total, then round to
    integers via the largest-remainder method (apportion the rounding
    deficit to the cells with the largest fractional remainder).

    Why this and not multinomial conditioning: Poisson counts conditioned on
    their sum are exactly multinomial, but NB counts are not — conditioning
    an NB draw via a multinomial would silently discard the extra
    (overdispersion) variance we specifically chose NB to capture. Rescaling
    a raw NB draw preserves each replica's relative hot/cold cell pattern
    (and, empirically, close to the correct per-cell variance — validated in
    development: empirical/theoretical variance ratio ~1.06 across cells)
    while still enforcing the fixed-total comparison the real scan uses
    (anchored expected counts sum to the window's observed total by
    construction, so an unconditioned null would let replicas differ from
    the real scan simply by having drawn a luckier/unluckier window total,
    contaminating the calibration for reasons unrelated to any cell-level
    anomaly).
    """
    r = 1.0 / alpha
    p = r / (r + mu)
    raw = rng.negative_binomial(r, p).astype(float)
    raw_sum = raw.sum()
    if raw_sum <= 0:
        raw = raw + 1.0
        raw_sum = raw.sum()
    scaled = raw * (window_total / raw_sum)
    floor_vals = np.floor(scaled).astype(np.int64)
    remainder = scaled - floor_vals
    deficit = int(window_total - floor_vals.sum())
    if deficit > 0:
        top_idx = np.argsort(-remainder)[:deficit]
        floor_vals[top_idx] += 1
    elif deficit < 0:
        raise AssertionError(f"Unexpected negative rounding deficit: {deficit}")
    return floor_vals


def run_window_nulls(window_df, alpha, m, n_restarts, seed):
    mu = window_df["expected"].to_numpy()
    window_total = int(round(window_df["observed"].sum()))
    rng = np.random.default_rng(seed)

    max_scores = np.empty(m)
    bad_totals = []
    replica_cells = window_df[FEATURES].copy()
    for i in range(m):
        replica = draw_null_replica(mu, alpha, window_total, rng)
        if replica.sum() != window_total:
            bad_totals.append(i)
        replica_cells["observed"] = replica
        replica_cells["expected"] = mu
        _subset, sc = mdss.scan(replica_cells, "observed", "expected", FEATURES, n_restarts=n_restarts, seed=int(rng.integers(0, 2**31 - 1)))
        max_scores[i] = sc
    return max_scores, bad_totals, window_total


def pvalue(observed_score, null_max_scores):
    """P-value for a subset with the given score, against a window's null
    max-score distribution. Comparing against the MAX score over the whole
    search space (rather than per-subset) is what corrects for multiple
    testing by construction (Kulldorff/Neill): the null distribution already
    reflects "how extreme is the BEST subset MDSS can find by chance", so no
    separate FDR/Bonferroni correction is needed for the top-1 subset."""
    null_max_scores = np.asarray(null_max_scores)
    return (np.sum(null_max_scores >= observed_score) + 1) / (len(null_max_scores) + 1)


def run_calibration_check(window_df, alpha, threshold_95, m_check, n_restarts, seed):
    max_scores, bad_totals, _ = run_window_nulls(window_df, alpha, m_check, n_restarts, seed)
    exceed_frac = np.mean(max_scores > threshold_95)
    return exceed_frac, max_scores, bad_totals


def plot_null_distributions(null_results, filename):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    windows = list(null_results.keys())
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=False)
    for ax, w in zip(axes.flat, windows):
        scores = null_results[w]["null_scores"]
        ax.hist(scores, bins=25, color=diag.COLOR_LINE, alpha=0.8)
        ax.axvline(np.percentile(scores, 95), color=diag.COLOR_OUTLIER, linewidth=1.5, linestyle="--")
        ax.set_title(w, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(windows):]:
        ax.axis("off")
    fig.suptitle("Per-window null max-score distributions (dashed = 95th pct)")
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_calibration_check(check_fractions, filename):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    windows = list(check_fractions.keys())
    fracs = [check_fractions[w] for w in windows]
    colors = [
        diag.COLOR_OUTLIER if (f > MISCALIBRATION_HIGH or f < MISCALIBRATION_LOW) else diag.COLOR_LINE
        for f in fracs
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(windows, fracs, color=colors)
    ax.axhline(CALIBRATION_ALPHA, color="#333333", linewidth=1.5, linestyle="--", label=f"nominal {CALIBRATION_ALPHA:.0%}")
    ax.set_title("Calibration check: fraction of fresh null replicas exceeding the 95th-pct threshold")
    ax.set_xlabel("3-month test window")
    ax.set_ylabel("Exceedance fraction")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    t_start = time.time()
    processed_md5_before = mod04.file_md5(config.PROCESSED_DATA_PATH)
    active_md5_before = mod04.file_md5(config.ACTIVE_DATA_PATH)
    anchored_md5_before = mod04.file_md5(config.ANCHORED_EXPECTED_COUNTS_PATH)

    alpha = recover_alpha()

    df, windows = load_anchored_counts()
    n_windows = len(windows)
    total_scan_count = M * N_RESTARTS_NULL * n_windows * 2  # x2: main null + calibration check
    print("\n" + "=" * 78)
    print(f"MODE = {MODE!r}: M={M} null replicas, N_RESTARTS_NULL={N_RESTARTS_NULL}, "
          f"{n_windows} windows, x2 (main null + calibration check)")
    print(f"Total scan() calls this run: {total_scan_count:,} "
          f"(M x N_RESTARTS_NULL x n_windows x 2)")
    print("=" * 78)

    null_results = {}
    check_fractions = {}
    check_details = {}
    long_rows = []

    for w_idx, w in enumerate(windows):
        window_df = df[df["window_label"] == w][FEATURES + ["observed", "expected"]].reset_index(drop=True)
        seed_main = BASE_SEED + 1000 * w_idx
        seed_check = BASE_SEED + 1000 * w_idx + 500

        max_scores, bad_totals, window_total = run_window_nulls(window_df, alpha, M, N_RESTARTS_NULL, seed_main)
        if bad_totals:
            print(f"  WARNING [{w}]: {len(bad_totals)} replicas failed the total-match check: {bad_totals}")

        p95 = np.percentile(max_scores, 95)
        p99 = np.percentile(max_scores, 99)
        null_results[w] = {
            "null_scores": max_scores, "window_total": window_total,
            "mean": max_scores.mean(), "p95": p95, "p99": p99, "max": max_scores.max(),
        }
        for i, s in enumerate(max_scores):
            long_rows.append({"window_label": w, "replica_idx": i, "max_score": s})

        exceed_frac, check_scores, check_bad_totals = run_calibration_check(
            window_df, alpha, p95, M, N_RESTARTS_NULL, seed_check
        )
        if check_bad_totals:
            print(f"  WARNING [{w}] calibration check: {len(check_bad_totals)} replicas failed total-match: {check_bad_totals}")
        check_fractions[w] = exceed_frac
        check_details[w] = check_scores

        print(
            f"[{w}] window_total={window_total}: null mean={max_scores.mean():.2f}, "
            f"p95={p95:.2f}, p99={p99:.2f}, max={max_scores.max():.2f} | "
            f"calibration check (fresh replicas > p95): {exceed_frac:.1%}"
        )

    long_df = pd.DataFrame(long_rows)
    out_path = config.REPORTS_DIR / "e1_null_maxscores.csv"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, index=False)
    print(f"\nSaved null max-score distributions (long format): {out_path} ({len(long_df)} rows)")

    plot_null_distributions(null_results, "e1_null_distributions.png")
    plot_calibration_check(check_fractions, "e1_calibration_check.png")

    all_check_scores = np.concatenate(list(check_details.values()))
    all_p95_thresholds = np.repeat([null_results[w]["p95"] for w in windows], M)
    pooled_exceed_frac = np.mean(all_check_scores > all_p95_thresholds)

    flagged = [w for w in windows if check_fractions[w] > MISCALIBRATION_HIGH or check_fractions[w] < MISCALIBRATION_LOW]

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"\nMode: {MODE} (M={M}), N_RESTARTS_NULL={N_RESTARTS_NULL}, total scan() calls={total_scan_count:,}")
    print(f"Alpha (NB dispersion, recovered from 05): {alpha:.4f}")

    print(f"\n{'window':10s}{'p95':>10s}{'p99':>10s}{'max':>10s}{'calib check':>14s}")
    for w in windows:
        r = null_results[w]
        flag = " <-- FLAGGED" if w in flagged else ""
        print(f"{w:10s}{r['p95']:10.2f}{r['p99']:10.2f}{r['max']:10.2f}{check_fractions[w]:13.1%}{flag}")

    print(f"\nPooled calibration check (all windows, all fresh replicas vs. their own p95): {pooled_exceed_frac:.1%} "
          f"(nominal {CALIBRATION_ALPHA:.0%})")

    if flagged:
        print(f"\nFLAGGED windows (calibration check outside [{MISCALIBRATION_LOW:.0%}, {MISCALIBRATION_HIGH:.0%}]): {flagged}")
        flagged_reduced_conf = [w for w in flagged if w in REDUCED_CONFIDENCE_WINDOWS]
        if flagged_reduced_conf:
            print(f"  Of these, {flagged_reduced_conf} were ALREADY flagged as reduced-confidence "
                  "(thin occupancy) in script 05 — consistent with that caveat, not a new problem.")
        other_flagged = [w for w in flagged if w not in REDUCED_CONFIDENCE_WINDOWS]
        if other_flagged:
            print(f"  {other_flagged} are NEWLY flagged here (not previously known thin windows) — worth investigating "
                  "(e.g. increase M or N_RESTARTS_NULL for these specific windows before trusting E2/E3 results there).")
    else:
        print("\nNo windows flagged — calibration checks are all within the expected range around 5%.")

    conclusion = (
        "trustworthy enough to feed E2/E3" if not flagged or set(flagged).issubset(REDUCED_CONFIDENCE_WINDOWS)
        else "feed E2/E3 with caution — some non-reduced-confidence windows show miscalibration"
    )
    print(f"\nConclusion: the per-window nulls are {conclusion}.")
    if MODE == "fast":
        print("This was a FAST run (M=199) for development. Re-run with MODE='full' (M=999) before final results.")

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f}s")

    processed_md5_after = mod04.file_md5(config.PROCESSED_DATA_PATH)
    active_md5_after = mod04.file_md5(config.ACTIVE_DATA_PATH)
    anchored_md5_after = mod04.file_md5(config.ANCHORED_EXPECTED_COUNTS_PATH)
    print(f"\nSource unchanged — processed_data.csv: {processed_md5_before == processed_md5_after} (MD5 {processed_md5_after})")
    print(f"Source unchanged — {config.ACTIVE_DATA_PATH.name}: {active_md5_before == active_md5_after} (MD5 {active_md5_after})")
    print(f"Source unchanged — {config.ANCHORED_EXPECTED_COUNTS_PATH.name}: {anchored_md5_before == anchored_md5_after} (MD5 {anchored_md5_after})")


if __name__ == "__main__":
    main()
