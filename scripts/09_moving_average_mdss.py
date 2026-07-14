"""Moving-average MDSS baseline, registered as a second method in 08's E2
plug-in harness.

THE ISOLATED VARIABLE: fit_expected re-anchors a per-cell SHARE to the
detection window's own total — structurally IDENTICAL to the proposed
method's mod08._mdss_fit_expected (`share * window_cells['observed'].sum()`).
The ONLY thing that differs is where the share comes from: a trailing
K-month EMPIRICAL average here, instead of the trained Poisson regression.
Absolute trailing counts were shown broken in 04 (27% under-prediction on
the rising period, wrong-signed trend); anchoring is applied uniformly to
both methods precisely so the E2 comparison isolates the share-estimation
mechanism, not staleness.

WHY THIS MODULE HAS ITS OWN ORCHESTRATION LOOP, NOT mod08.run_step_experiment:
mod08's Method.fit_expected is a ONE-argument callable (window_cells) — the
proposed method's share is fixed/precomputed once from training, so a single
argument suffices. The task spec for this baseline is explicitly
fit_expected(detection_window, reference_data) — TWO arguments — because a
statistically honest MA calibration needs a FRESH trailing-reference draw
per trial (not one reused across all trials for a window: that would
understate MA's true null variance, since a real deployment sees different
recent history every time too). A one-argument interface cannot carry that
per-trial reference through mod08's existing loop without a closure hack
that reaches into 08 internals. So: 08's FILE IS NOT MODIFIED (verified via
MD5 below) and NONE of its logic is duplicated — every shared primitive
(scan, pvalue, draw_null_replica, inject, draw_subgroup, jaccard, recall,
deterministic_seed, ramp_multipliers, valid_onset_windows, the Method
dataclass, and even mod08._mdss_detect itself, reused verbatim as MA's own
detect — same score+pvalue logic, only fit_expected differs) is imported
from 01/06/07/08. Only the per-trial reference-building and the orchestration
loop that needs the second argument are new, and they are built to emit rows
in the EXACT same schema as 08's CSVs so they can be appended directly.

RAMP'S DELIBERATE ASYMMETRY WITH STEP:
STEP always draws a FRESH, clean null replica for the trailing reference
(no injection leakage) — isolating "can MA detect a NEW anomaly" from any
reference contamination. RAMP does the opposite ON PURPOSE: as the ramp
progresses, each step's trailing reference is built from the PRECEDING
ramp steps' own (already-injected) realized counts within that same trial.
This lets the rising signal leak into MA's own baseline over time — the
"boiling frog" effect, a genuine, reportable weakness of moving-average
detectors under a sustained trend, not a bug.

Read-only on config.ACTIVE_DATA_PATH, config.PROCESSED_DATA_PATH,
config.ANCHORED_EXPECTED_COUNTS_PATH, reports/e1_null_maxscores.csv, and the
existing E2 CSVs (appended to, never rewritten in place without the prior
rows). Imports scan()/pvalue()/draw_null_replica() rather than reimplementing.
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
e1 = _load_module("e1_07", _scripts_dir / "07_e1_calibration.py")
mod08 = _load_module("mod08", _scripts_dir / "08_e2_injection.py")

# Reuse the proposed method's exact grid/trial/seed settings — required for
# a fair, same-seed, same-grid head-to-head comparison.
FEATURES = mod08.FEATURES
J_GRID = mod08.J_GRID
R_GRID = mod08.R_GRID
N_TRIALS_FULL = mod08.N_TRIALS_FULL
N_TRIALS_GATE = mod08.N_TRIALS_GATE
N_RESTARTS_GRID = mod08.N_RESTARTS_GRID
N_RESTARTS_GATE = mod08.N_RESTARTS_GATE
RAMP_LENGTH = mod08.RAMP_LENGTH
MIN_BASELINE_EXPECTED = mod08.MIN_BASELINE_EXPECTED
BASE_SEED = mod08.BASE_SEED
FLAG_WINDOWS = mod08.FLAG_WINDOWS
deterministic_seed = mod08.deterministic_seed
draw_subgroup = mod08.draw_subgroup
jaccard = mod08.jaccard
recall = mod08.recall
flatten_subset = mod08.flatten_subset
inject = mod08.inject
ramp_multipliers = mod08.ramp_multipliers
valid_onset_windows = mod08.valid_onset_windows
Method = mod08.Method

METHOD_NAME = "moving_average_mdss"

# ---------------------------------------------------------------------------
# FAST/FULL toggle
# ---------------------------------------------------------------------------
MODE = "fast"
M_BY_MODE = {"fast": 199, "full": 999}
M_CALIBRATION = M_BY_MODE[MODE]
N_RESTARTS_NULL = 3  # matches 07's null-replica restart count, same rationale
K_CANDIDATES_MONTHS = [3, 6]


def precision(recovered_subset, true_subset):
    """Fraction of the RECOVERED (feature, value) pairs that are actually
    true. Reported alongside recall because recall alone is gameable by
    over-selection (restrict everything -> recall trivially high); precision
    exposes exactly that tendency."""
    recovered_pairs = flatten_subset(recovered_subset)
    if not recovered_pairs:
        return float("nan")
    true_pairs = flatten_subset(true_subset)
    return len(true_pairs & recovered_pairs) / len(recovered_pairs)


# =============================================================================
# SETUP / REFERENCE INFRASTRUCTURE
# =============================================================================

def build_infrastructure():
    """One-time setup shared by calibration, gate, and the injection grid:
    the trained share regression (for the null MEAN used only to draw
    synthetic reference replicas — MA's own fit_expected never sees this
    regression directly, only its output via null draws), alpha, the
    canonical 613-cell frame (same row order as mod08's base_data, verified
    empirically identical across windows), training-level scoreability, and
    real per-window totals extended back into the 2021 buffer (needed
    because a K=6 trailing reference for the earliest test windows reaches
    into the buffer period)."""
    df = diag.load_data(config.ACTIVE_DATA_PATH)
    train_df, _test_df, train_levels = mod05.prepare_data(df)
    bundle = mod05.fit_share_model(train_df)

    base_data, null_scores_by_window, windows = mod08.load_base_data()
    cells_613 = base_data[windows[0]][FEATURES].reset_index(drop=True)

    month_idx, window_idx = diag.build_time_indices(df)
    df["window_idx"] = window_idx
    for c in FEATURES:
        df[c] = df[c].astype(str)
    cells_613_str = cells_613.copy()
    cells_613_str["syndicate"] = cells_613_str["syndicate"].astype(str)
    key_613 = set(map(tuple, cells_613_str[FEATURES].values))
    df_key = df[FEATURES].apply(tuple, axis=1)
    in_613 = df_key.isin(key_613)

    window_totals = {}
    for w_idx in range(12, 16):  # 2021 buffer, restricted to the 613-cell universe
        window_totals[w_idx] = int(df[(df["window_idx"] == w_idx) & in_613].shape[0])
    for w_idx, w in zip(range(16, 28), windows):  # test period: reuse the anchored CSV's own totals
        window_totals[w_idx] = int(round(base_data[w]["observed"].sum()))

    scoreable = cells_613.apply(lambda r: all(str(r[c]) in train_levels[c] for c in FEATURES), axis=1).to_numpy()

    return {
        "pois": bundle["pois"], "alpha": bundle["alpha"], "cells_613": cells_613,
        "scoreable": scoreable, "window_totals": window_totals,
        "base_data": base_data, "null_scores_by_window": null_scores_by_window, "windows": windows,
    }


def window_idx_of(window_label, infra):
    return int(infra["base_data"][window_label]["window_idx"].iloc[0])


def mu_hat_for_window(window_idx, infra):
    """Regression rate per cell (unanchored) for window_idx's calendar
    quarter. Backoff for training-unscoreable cells mirrors 05/08: floor at
    the minimum rate among scoreable cells (same quarter), not a uniform
    +1 that would dilute ~530 already-known cells."""
    quarter = window_idx % 4
    cells = infra["cells_613"]
    scoreable = infra["scoreable"]
    mu = np.zeros(len(cells))
    tmp = cells.loc[scoreable].copy()
    tmp["syndicate"] = tmp["syndicate"].astype(str)  # regression was trained with syndicate as str (mod05.prepare_data)
    tmp["quarter"] = quarter
    mu[scoreable] = infra["pois"].predict(tmp).to_numpy()
    if (~scoreable).any():
        floor = mu[scoreable].min() if scoreable.any() else 1e-3
        mu[~scoreable] = floor
    return mu


def draw_reference_replica_df(window_idx, infra, rng):
    """A single NB2 null replica for window_idx (any window 12-27), as a
    DataFrame (FEATURES + 'observed'), so it composes directly with
    mod08.inject()/mod08's cell-masking utilities."""
    mu_raw = mu_hat_for_window(window_idx, infra)
    total = infra["window_totals"][window_idx]
    anchored_mu = mu_raw / mu_raw.sum() * total
    counts = e1.draw_null_replica(anchored_mu, infra["alpha"], total, rng)
    out = infra["cells_613"].copy()
    out["observed"] = counts
    return out


def pool_to_share(count_dfs):
    """Sum observed counts across the reference DataFrames (same cell order
    guaranteed by construction), backoff-floor zero cells at the minimum
    nonzero pooled count (mirrors 05's asymmetric floor — only dilutes the
    genuinely-absent cells, not the whole distribution), normalize to a
    share vector aligned to infra['cells_613']'s row order."""
    pooled = np.sum([d["observed"].to_numpy() for d in count_dfs], axis=0).astype(float)
    nonzero = pooled[pooled > 0]
    floor = nonzero.min() if len(nonzero) else 1.0
    pooled_floored = np.where(pooled > 0, pooled, floor)
    return pooled_floored / pooled_floored.sum()


def ma_fit_expected(window_cells, trailing_share):
    """THE ISOLATED VARIABLE, made explicit: identical re-anchoring formula
    to mod08._mdss_fit_expected (share * current total); trailing_share is
    the only thing that differs from the proposed method's fixed,
    regression-derived share."""
    total = window_cells["observed"].sum()
    return trailing_share * total


ma_detect = mod08._mdss_detect  # reused verbatim: same scan()+pvalue() logic, zero duplication


def trailing_share_clean(detection_window_idx, k_windows, infra, rng):
    """STEP/calibration reference: k_windows FRESH clean null replicas drawn
    from the calendar windows strictly preceding detection_window_idx."""
    ref_dfs = [draw_reference_replica_df(detection_window_idx - b, infra, rng) for b in range(1, k_windows + 1)]
    return pool_to_share(ref_dfs)


# =============================================================================
# MA CALIBRATION (own null, per K)
# =============================================================================

def calibrate_ma(K_months, infra, m, n_restarts, seed_base, verbose=True):
    k_windows = K_months // 3
    null_results, check_fractions, long_rows = {}, {}, []

    for w_idx_pos, w in enumerate(infra["windows"]):
        w_idx = window_idx_of(w, infra)
        mu_det = mu_hat_for_window(w_idx, infra)
        total = infra["window_totals"][w_idx]
        anchored_mu_det = mu_det / mu_det.sum() * total

        def one_replica(seed):
            rng = np.random.default_rng(seed)
            det_counts = e1.draw_null_replica(anchored_mu_det, infra["alpha"], total, rng)
            det_df = infra["cells_613"].copy()
            det_df["observed"] = det_counts
            share = trailing_share_clean(w_idx, k_windows, infra, rng)
            expected = ma_fit_expected(det_df, share)
            cells = det_df[FEATURES].copy()
            cells["observed"] = det_df["observed"].to_numpy()
            cells["expected"] = expected
            _sub, score = mdss.scan(cells, "observed", "expected", FEATURES, n_restarts=n_restarts, seed=int(rng.integers(0, 2**31 - 1)))
            return score

        max_scores = np.array([one_replica(seed_base + K_months * 100_000 + w_idx_pos * 10_000 + i) for i in range(m)])
        p95 = np.percentile(max_scores, 95)
        null_results[w] = {"scores": max_scores, "p95": p95, "p99": np.percentile(max_scores, 99), "max": max_scores.max()}
        for i, s in enumerate(max_scores):
            long_rows.append({"K": K_months, "window_label": w, "replica_idx": i, "max_score": s})

        check_scores = np.array([one_replica(seed_base + K_months * 100_000 + w_idx_pos * 10_000 + 50_000 + i) for i in range(m)])
        exceed_frac = np.mean(check_scores > p95)
        check_fractions[w] = exceed_frac
        if verbose:
            print(f"  [K={K_months}][{w}] p95={p95:.2f}, calibration check={exceed_frac:.1%}")

    return null_results, check_fractions, pd.DataFrame(long_rows)


def select_K(infra, m, n_restarts, seed_base):
    print("=" * 78)
    print("REFERENCE-LENGTH SELECTION: K=3 vs K=6 months")
    print("=" * 78)
    results = {}
    for K in K_CANDIDATES_MONTHS:
        print(f"\nCalibrating MA at K={K} months ({K // 3} trailing window(s))...")
        null_results, check_fractions, long_df = calibrate_ma(K, infra, m, n_restarts, seed_base)
        pooled_fpr = np.mean(list(check_fractions.values()))
        results[K] = {"null_results": null_results, "check_fractions": check_fractions, "long_df": long_df, "pooled_fpr": pooled_fpr}
        print(f"K={K}: pooled calibration FPR = {pooled_fpr:.1%} (target 5%)")

    chosen_K = min(K_CANDIDATES_MONTHS, key=lambda K: abs(results[K]["pooled_fpr"] - 0.05))
    fpr_summary = ", ".join(f"{K}:{results[K]['pooled_fpr']:.1%}" for K in K_CANDIDATES_MONTHS)
    print(f"\nChosen K = {chosen_K} months (pooled FPR {results[chosen_K]['pooled_fpr']:.1%} is closest to 5% among {fpr_summary})")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([results[K]["long_df"] for K in K_CANDIDATES_MONTHS], ignore_index=True)
    out_path = config.REPORTS_DIR / "e2_ma_null_maxscores.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(combined)} rows, both K values)")

    return chosen_K, results


# =============================================================================
# STEP experiment (MA)
# =============================================================================

def run_ma_step_trial(window_df_base, S, r, k_windows, infra, null_scores, n_restarts, trial_seed):
    rng_det = np.random.default_rng(trial_seed)
    w_idx = int(window_df_base["window_idx"].iloc[0]) if "window_idx" in window_df_base.columns else None
    # detection window: fresh clean null replica, using the REGRESSION mean anchored to this window's real total
    mu_det = mu_hat_for_window(w_idx, infra)
    total = infra["window_totals"][w_idx]
    anchored_mu_det = mu_det / mu_det.sum() * total
    clean_counts = e1.draw_null_replica(anchored_mu_det, infra["alpha"], total, rng_det)
    clean_df = infra["cells_613"].copy()
    clean_df["observed"] = clean_counts

    rng_ref_clean = np.random.default_rng(trial_seed + 1)
    share_clean = trailing_share_clean(w_idx, k_windows, infra, rng_ref_clean)
    exp_clean = ma_fit_expected(clean_df, share_clean)
    clean_det, clean_sub, clean_sc, clean_p = ma_detect(clean_df, pd.Series(exp_clean, index=clean_df.index), null_scores, n_restarts, trial_seed + 2)

    injected_df = inject(clean_df, S, r)
    rng_ref_inj = np.random.default_rng(trial_seed + 3)
    share_inj = trailing_share_clean(w_idx, k_windows, infra, rng_ref_inj)  # fresh, independent — no leakage
    exp_inj = ma_fit_expected(injected_df, share_inj)
    inj_det, inj_sub, inj_sc, inj_p = ma_detect(injected_df, pd.Series(exp_inj, index=injected_df.index), null_scores, n_restarts, trial_seed + 4)

    return (clean_det, clean_sub, clean_sc, clean_p), (inj_det, inj_sub, inj_sc, inj_p)


def run_ma_step_experiment(infra, ma_null_scores_by_window, k_windows, j_grid, r_grid, n_trials, n_restarts, seed_base, min_baseline_expected):
    rows = []
    pooled_ref = pd.concat(infra["base_data"].values(), ignore_index=True).groupby(FEATURES, as_index=False)["expected"].sum()

    for j in j_grid:
        for r in r_grid:
            cell_seed = seed_base + deterministic_seed(METHOD_NAME, j, r) % 10_000
            rng_subgroups = np.random.default_rng(cell_seed)

            for trial_idx in range(n_trials):
                S, _ = draw_subgroup(rng_subgroups, j, pooled_ref, min_baseline_expected)
                if S is None:
                    continue
                for w in infra["windows"]:
                    window_df = infra["base_data"][w]
                    mask = pd.Series(True, index=window_df.index)
                    for f, vals in S.items():
                        mask &= window_df[f].isin(vals)
                    if window_df.loc[mask, "expected"].sum() < min_baseline_expected:
                        continue

                    trial_seed = seed_base + deterministic_seed(METHOD_NAME, j, r, trial_idx, w) % 1_000_000
                    window_df_tagged = window_df.copy()
                    window_df_tagged["window_idx"] = window_idx_of(w, infra)

                    (clean_det, _cs, clean_sc, clean_p), (inj_det, inj_sub, inj_sc, inj_p) = run_ma_step_trial(
                        window_df_tagged, S, r, k_windows, infra, ma_null_scores_by_window[w], n_restarts, trial_seed
                    )
                    rows.append({
                        "method": METHOD_NAME, "j": j, "r": r, "window": w, "trial": trial_idx,
                        "arm": "clean", "S": repr(S), "detected": clean_det, "score": clean_sc,
                        "pvalue": clean_p, "jaccard": np.nan, "recall": np.nan, "precision": np.nan,
                    })
                    rows.append({
                        "method": METHOD_NAME, "j": j, "r": r, "window": w, "trial": trial_idx,
                        "arm": "injected", "S": repr(S), "detected": inj_det, "score": inj_sc,
                        "pvalue": inj_p, "jaccard": jaccard(inj_sub, S), "recall": recall(inj_sub, S),
                        "precision": precision(inj_sub, S),
                    })
    return pd.DataFrame(rows)


def summarize_ma_step(step_df):
    summaries = []
    for (method, j, r), g in step_df.groupby(["method", "j", "r"]):
        clean = g[g["arm"] == "clean"]
        injected = g[g["arm"] == "injected"]
        detected = injected[injected["detected"]]
        summaries.append({
            "method": method, "j": j, "r": r,
            "n_clean": len(clean), "fpr": clean["detected"].mean(),
            "n_injected": len(injected), "power": injected["detected"].mean(),
            "mean_jaccard_detected": detected["jaccard"].mean(), "mean_jaccard_all": injected["jaccard"].mean(),
            "mean_recall_detected": detected["recall"].mean(), "mean_recall_all": injected["recall"].mean(),
            "mean_precision_detected": detected["precision"].mean(), "mean_precision_all": injected["precision"].mean(),
        })
    return pd.DataFrame(summaries)


# =============================================================================
# RAMP experiment (MA) — boiling-frog leakage by design
# =============================================================================

def run_ma_ramp_trial(S, r, k_windows, onset_idx, infra, ma_null_scores_by_window, n_restarts, seed_base):
    windows = infra["windows"]
    multipliers = ramp_multipliers(r, RAMP_LENGTH)
    ramp_series = []  # realized (post-injection) DataFrames for steps already run, for future leakage
    detected_at = None

    for step, mult in enumerate(multipliers):
        w = windows[onset_idx + step]
        w_idx = window_idx_of(w, infra)
        step_seed = seed_base + step * 10

        rng_det = np.random.default_rng(step_seed)
        mu_det = mu_hat_for_window(w_idx, infra)
        total = infra["window_totals"][w_idx]
        anchored_mu_det = mu_det / mu_det.sum() * total
        clean_counts = e1.draw_null_replica(anchored_mu_det, infra["alpha"], total, rng_det)
        clean_df = infra["cells_613"].copy()
        clean_df["observed"] = clean_counts
        injected_df = inject(clean_df, S, mult)

        # trailing reference: leak in the PRIOR (already-injected) ramp steps; fall
        # back to fresh clean draws for any depth reaching before ramp onset.
        rng_ref = np.random.default_rng(step_seed + 1)
        ref_dfs = []
        for back in range(1, k_windows + 1):
            src_step = step - back
            if src_step >= 0:
                ref_dfs.append(ramp_series[src_step])
            else:
                ref_dfs.append(draw_reference_replica_df(w_idx - back, infra, rng_ref))
        share = pool_to_share(ref_dfs)
        expected = ma_fit_expected(injected_df, share)

        det, sub, sc, p = ma_detect(injected_df, pd.Series(expected, index=injected_df.index), ma_null_scores_by_window[w], n_restarts, step_seed + 2)
        if det and detected_at is None:
            detected_at = step + 1
        ramp_series.append(injected_df)

    return detected_at, windows[onset_idx]


def run_ma_ramp_experiment(infra, ma_null_scores_by_window, k_windows, j_grid, r_grid, n_trials, n_restarts, seed_base, min_baseline_expected):
    onset_candidates = valid_onset_windows(infra["windows"], RAMP_LENGTH)
    rows = []
    pooled_ref = pd.concat(infra["base_data"].values(), ignore_index=True).groupby(FEATURES, as_index=False)["expected"].sum()

    for j in j_grid:
        for r in r_grid:
            cell_seed = seed_base + deterministic_seed("ramp", METHOD_NAME, j, r) % 10_000
            rng_subgroups = np.random.default_rng(cell_seed)
            rng_onset = np.random.default_rng(cell_seed + 1)

            for trial_idx in range(n_trials):
                S, _ = draw_subgroup(rng_subgroups, j, pooled_ref, min_baseline_expected)
                if S is None:
                    continue
                onset_window = onset_candidates[rng_onset.integers(0, len(onset_candidates))]
                onset_idx = infra["windows"].index(onset_window)
                trial_seed = seed_base + deterministic_seed(METHOD_NAME, j, r, trial_idx, onset_window) % 1_000_000

                detected_at, onset_w = run_ma_ramp_trial(S, r, k_windows, onset_idx, infra, ma_null_scores_by_window, n_restarts, trial_seed)
                rows.append({
                    "method": METHOD_NAME, "j": j, "r": r, "trial": trial_idx,
                    "onset_window": onset_w, "S": repr(S),
                    "time_to_detect": detected_at if detected_at is not None else np.nan,
                    "censored": detected_at is None,
                })
    return pd.DataFrame(rows)


def summarize_ma_ramp(ramp_df):
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

def run_validation_gate_ma(infra, ma_null_scores_by_window, chosen_K, n_restarts_gate=N_RESTARTS_GATE):
    print("\n" + "=" * 78)
    print("VALIDATION GATE (MA, before the full grid)")
    print("=" * 78)
    k_windows = chosen_K // 3
    checks = {}

    gate_infra = {**infra, "windows": ["2023-Q3"], "base_data": {"2023-Q3": infra["base_data"]["2023-Q3"]}}
    gate_null = {"2023-Q3": ma_null_scores_by_window["2023-Q3"]}
    step_df = run_ma_step_experiment(
        gate_infra, gate_null, k_windows, j_grid=[2], r_grid=[1.5, 3.0], n_trials=N_TRIALS_GATE, n_restarts=n_restarts_gate,
        seed_base=BASE_SEED + 500, min_baseline_expected=MIN_BASELINE_EXPECTED,
    )
    summary = summarize_ma_step(step_df)
    print(summary.to_string(index=False))

    fpr = summary["fpr"].mean()
    fpr_ok = 0.0 <= fpr <= 0.20
    checks["clean_fpr_near_5pct"] = fpr_ok
    print(f"\n[{'PASS' if fpr_ok else 'FAIL'}] Clean-arm FPR = {fpr:.1%}")

    power_15 = summary.loc[summary["r"] == 1.5, "power"].iloc[0]
    power_30 = summary.loc[summary["r"] == 3.0, "power"].iloc[0]
    power_rises = power_30 > power_15
    checks["power_rises_with_r"] = power_rises
    print(f"[{'PASS' if power_rises else 'FAIL'}] Power rises with r: power(1.5)={power_15:.2f} -> power(3.0)={power_30:.2f}")

    mean_recall_strong = summary.loc[summary["r"] == 3.0, "mean_recall_detected"].iloc[0]
    recall_ok = pd.notna(mean_recall_strong) and mean_recall_strong >= 0.5
    checks["recovers_planted_subgroup"] = recall_ok
    print(f"[{'PASS' if recall_ok else 'FAIL'}] Mean recall on detected strong-signal trials = {mean_recall_strong:.2f} (>=0.5 required)")

    all_pass = all(checks.values())
    print(f"\nGate result: {'ALL PASS' if all_pass else 'FAILED — STOPPING'}")
    return all_pass, checks


# =============================================================================
# PLOTTING (method-agnostic: groups by whatever 'method' values are present)
# =============================================================================

def plot_power_curves_by_method(step_summary, filename):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {}
    colors = [diag.COLOR_LINE, diag.COLOR_OUTLIER, "#5B8C5A", "#8862AE"]
    for i, method in enumerate(sorted(step_summary["method"].unique())):
        for j in sorted(step_summary["j"].unique()):
            g = step_summary[(step_summary["method"] == method) & (step_summary["j"] == j)].sort_values("r")
            ls = "-" if j == min(step_summary["j"]) else "--"
            ax.plot(g["r"], g["power"], marker="o", linestyle=ls, color=colors[i % len(colors)], label=f"{method}, j={j}")
    ax.axhline(0.05, color="#333333", linewidth=1.5, linestyle=":", label="nominal FPR (5%)")
    ax.set_xlabel("injected multiplier r")
    ax.set_ylabel("detection power")
    ax.set_title("Power vs. injected multiplier, by method (step injection)")
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_precision_recall_jaccard(step_summary, filename):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    cols = ["mean_precision_detected", "mean_recall_detected", "mean_jaccard_detected"]
    titles = ["Precision (detected)", "Recall (detected)", "Jaccard (detected)"]
    colors = [diag.COLOR_LINE, diag.COLOR_OUTLIER, "#5B8C5A", "#8862AE"]
    for ax, col, title in zip(axes, cols, titles):
        for i, method in enumerate(sorted(step_summary["method"].unique())):
            for j in sorted(step_summary["j"].unique()):
                g = step_summary[(step_summary["method"] == method) & (step_summary["j"] == j)].sort_values("r")
                ls = "-" if j == min(step_summary["j"]) else "--"
                ax.plot(g["r"], g[col], marker="o", linestyle=ls, color=colors[i % len(colors)], label=f"{method}, j={j}")
        ax.set_xlabel("injected multiplier r")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("mean value")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_time_to_detect_by_method(ramp_summary, filename):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [diag.COLOR_LINE, diag.COLOR_OUTLIER, "#5B8C5A", "#8862AE"]
    for i, method in enumerate(sorted(ramp_summary["method"].unique())):
        for j in sorted(ramp_summary["j"].unique()):
            g = ramp_summary[(ramp_summary["method"] == method) & (ramp_summary["j"] == j)].sort_values("r")
            ls = "-" if j == min(ramp_summary["j"]) else "--"
            ax.plot(g["r"], g["median_time_to_detect"], marker="o", linestyle=ls, color=colors[i % len(colors)], label=f"{method}, j={j}")
    ax.set_xlabel("target multiplier r (ramp endpoint)")
    ax.set_ylabel("median time-to-detect (windows since onset)")
    ax.set_title("Time-to-detect vs. ramp target, by method")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
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
    md5_paths = {
        "processed_data.csv": config.PROCESSED_DATA_PATH,
        config.ACTIVE_DATA_PATH.name: config.ACTIVE_DATA_PATH,
        config.ANCHORED_EXPECTED_COUNTS_PATH.name: config.ANCHORED_EXPECTED_COUNTS_PATH,
        "e1_null_maxscores.csv": config.REPORTS_DIR / "e1_null_maxscores.csv",
    }
    md5_before = {name: mod04.file_md5(path) for name, path in md5_paths.items()}
    mod08_path = _scripts_dir / "08_e2_injection.py"
    mod08_md5_before = mod04.file_md5(mod08_path)
    step_csv = config.REPORTS_DIR / "e2_step_results.csv"
    ramp_csv = config.REPORTS_DIR / "e2_ramp_results.csv"
    prior_step_df = pd.read_csv(step_csv)
    prior_ramp_df = pd.read_csv(ramp_csv)

    print("Building infrastructure (regression, alpha, 613-cell frame, extended window totals)...")
    infra = build_infrastructure()
    print(f"Loaded {len(infra['windows'])} test windows; {infra['scoreable'].sum()}/{len(infra['scoreable'])} cells scoreable by the regression.")

    chosen_K, K_results = select_K(infra, M_CALIBRATION, N_RESTARTS_NULL, BASE_SEED + 700)
    ma_null_scores_by_window = {w: r["scores"] for w, r in K_results[chosen_K]["null_results"].items()}

    gate_pass, gate_checks = run_validation_gate_ma(infra, ma_null_scores_by_window, chosen_K)
    if not gate_pass:
        print("\nSTOPPING: MA validation gate failed. Not launching the full grid.")
        return

    k_windows = chosen_K // 3
    n_windows = len(infra["windows"])
    n_step_calls = len(J_GRID) * len(R_GRID) * N_TRIALS_FULL * n_windows * 2
    n_ramp_calls = len(J_GRID) * len(R_GRID) * N_TRIALS_FULL * RAMP_LENGTH
    print(f"\nMODE={MODE!r}, chosen K={chosen_K}. Total scan() calls (before per-window skips): "
          f"{n_step_calls + n_ramp_calls:,}, at n_restarts={N_RESTARTS_GRID}")

    print("\n" + "=" * 78)
    print("MA STEP INJECTION EXPERIMENT")
    print("=" * 78)
    ma_step_df = run_ma_step_experiment(infra, ma_null_scores_by_window, k_windows, J_GRID, R_GRID, N_TRIALS_FULL, N_RESTARTS_GRID, BASE_SEED + 800, MIN_BASELINE_EXPECTED)
    ma_step_summary = summarize_ma_step(ma_step_df)
    print(ma_step_summary.to_string(index=False))

    print("\n" + "=" * 78)
    print("MA RAMP INJECTION EXPERIMENT")
    print("=" * 78)
    ma_ramp_df = run_ma_ramp_experiment(infra, ma_null_scores_by_window, k_windows, J_GRID, R_GRID, N_TRIALS_FULL, N_RESTARTS_GRID, BASE_SEED + 900, MIN_BASELINE_EXPECTED)
    ma_ramp_summary = summarize_ma_ramp(ma_ramp_df)
    print(ma_ramp_summary.to_string(index=False))

    combined_step = pd.concat([prior_step_df, ma_step_df], ignore_index=True)
    combined_ramp = pd.concat([prior_ramp_df, ma_ramp_df], ignore_index=True)
    combined_step.to_csv(step_csv, index=False)
    combined_ramp.to_csv(ramp_csv, index=False)
    print(f"\nAppended MA rows: {step_csv} now {len(combined_step)} rows; {ramp_csv} now {len(combined_ramp)} rows")

    # The proposed method's summary was already computed (08 didn't collect
    # precision, since that metric was added for this baseline comparison) —
    # reuse it rather than recomputing, and pad the missing columns with NaN.
    prior_step_summary = pd.read_csv(config.REPORTS_DIR / "e2_step_summary.csv")
    for col in ["mean_precision_detected", "mean_precision_all"]:
        if col not in prior_step_summary.columns:
            prior_step_summary[col] = np.nan
    combined_step_summary = pd.concat([prior_step_summary, ma_step_summary], ignore_index=True)

    prior_ramp_summary = pd.read_csv(config.REPORTS_DIR / "e2_ramp_summary.csv")
    combined_ramp_summary = pd.concat([prior_ramp_summary, ma_ramp_summary], ignore_index=True)
    combined_step_summary.to_csv(config.REPORTS_DIR / "e2_step_summary.csv", index=False)
    combined_ramp_summary.to_csv(config.REPORTS_DIR / "e2_ramp_summary.csv", index=False)

    plot_power_curves_by_method(combined_step_summary, "e2_power_curves_by_method.png")
    plot_precision_recall_jaccard(combined_step_summary, "e2_precision_recall_jaccard_by_method.png")
    plot_time_to_detect_by_method(combined_ramp_summary, "e2_time_to_detect_by_method.png")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"\nChosen K = {chosen_K} months. Calibration FPRs: " +
          ", ".join(f"K={K}: {K_results[K]['pooled_fpr']:.1%}" for K in K_CANDIDATES_MONTHS))
    print(f"\nGate: {'PASS' if gate_pass else 'FAIL'} ({gate_checks})")
    print(f"\nMA step summary:\n{ma_step_summary.to_string(index=False)}")
    print(f"\nMA ramp summary:\n{ma_ramp_summary.to_string(index=False)}")

    print("\nHead-to-head (power, by j,r):")
    compare = combined_step_summary.pivot_table(index=["j", "r"], columns="method", values="power")
    print(compare.to_string())

    per_window_power = ma_step_df[ma_step_df["arm"] == "injected"].groupby("window")["detected"].mean()
    flagged = per_window_power[per_window_power.index.isin(FLAG_WINDOWS)]
    other = per_window_power[~per_window_power.index.isin(FLAG_WINDOWS)]
    print(f"\nMA pooled power (non-flagged windows): {other.mean():.2f}")
    print(f"MA power in flagged windows: {flagged.to_dict()}")

    mod08_md5_after = mod04.file_md5(mod08_path)
    print(f"\nHarness core (08_e2_injection.py) unchanged: {mod08_md5_before == mod08_md5_after}")

    elapsed = time.time() - t_start
    print(f"\nTotal compute: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print("\nSource artifact integrity:")
    for name, before in md5_before.items():
        after = mod04.file_md5(md5_paths[name])
        print(f"  {name}: unchanged={before == after}")


if __name__ == "__main__":
    main()
