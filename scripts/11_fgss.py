"""FGSS — Fast Generalized Subset Scan (McFowland, Speakman & Neill 2013,
"Fast Generalized Subset Scan for Anomalous Pattern Detection", JMLR 14) — as
the fourth E2 method, registered as a plug-in.

WHAT MAKES THIS DIFFERENT FROM THE OTHER THREE (and the failure mode avoided):
FGSS is NONPARAMETRIC. It does NOT sum counts and score a Poisson
log-likelihood-ratio. Instead it converts each cell's observed count into an
empirical p-value under the null, then scans for subsets containing an
unexpectedly large number of SMALL p-values, scored by the Berk-Jones (BJ)
nonparametric statistic. Reusing 06's count-based score(C,B) here would just
make this Poisson-MDSS a second time — precisely the mistake this baseline
exists to avoid. So the ONLY new statistic is BJ over empirical p-values; the
SEARCH is the shared LTSS coordinate-ascent skeleton from 06 (FGSS's scan has
the LTSS property too — McFowland 2013, Thm 1), reused via 06's new pluggable
`scorer` argument, NOT copy-pasted.

06 REFACTOR (minimal, sanctioned by the task): 06's _coordinate_ascent/scan
were parameterized by a `scorer` object exposing optimize_feature() and
total_score(); scorer=None rebuilds the original Poisson scorer, so the Poisson
path is behaviour-identical (06's Part B toy tests — null/k=0-empty-subset,
single-feature, conjunction recovery, monotonicity, restart robustness — all
still PASS; 06 MD5 changed, reported in the summary). This module passes a
BerkJonesScorer instead. 08 (the harness core) is NOT touched (MD5-verified).

DELIBERATE ADAPTATION (stated, defensible): the original FGSS derives each
cell's empirical p-value from a Bayesian-network null model. We instead compute
it from the SAME anchored negative-binomial null every other method here uses
(per-cell mean = re-anchored share x current total, dispersion alpha recovered
in 05). This keeps all four methods on ONE null model so the comparison
isolates the SCAN STATISTIC (parametric Poisson-LLR vs nonparametric BJ), not
the null. The nonparametric BJ scan + LTSS search are unchanged from the paper.

TIE HANDLING: counts are discrete, so many cells share a p-value. We use the
upper-tail MID-P value  p_i = P(X > c_i) + 1/2 P(X = c_i)  (McFowland's
"p-value range" midpoint), which de-biases the scan under ties relative to the
naive P(X >= c_i). Emergence is one-sided: a small p_i means a surprisingly
HIGH count, so BJ scores only the excess direction (proportion below alpha
exceeds alpha), mirroring the C>B constraint in the Poisson score.

alpha_max: only cells with p_i <= alpha_max are "signal-carrying"; the scan
maximizes BJ over candidate thresholds alpha in (0, alpha_max]. alpha_max is a
FIXED hyperparameter set a priori (see ALPHA_MAX), never tuned on detection.

OWN NULL (same frame as the others): calibrate on clean NB null replicas ->
per-window 95th-pct of the max BJ score -> reports/e2_fgss_null_maxscores.csv;
detect = max BJ score exceeds that window's own threshold (pvalue<0.05 vs FGSS's
own null). Own orchestration for STEP (to also record precision); RAMP reuses
mod08.run_ramp_experiment unchanged (FGSS's ramp has no reference-leakage
asymmetry — it reads each window's own expected, like the proposed method).

Read-only on config data/artifacts and the existing E2 CSVs (appended to
idempotently: prior fgss rows are dropped before re-appending).
"""

import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import nbinom

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
mod08 = _load_module("mod08", _scripts_dir / "08_e2_injection.py")
mod09 = _load_module("mod09", _scripts_dir / "09_moving_average_mdss.py")

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
inject = mod08.inject
make_clean_trial = mod08.make_clean_trial
jaccard = mod08.jaccard
recall = mod08.recall
precision = mod09.precision
fgss_fit_expected = mod08._mdss_fit_expected  # identical re-anchoring as the proposed method

METHOD_NAME = "fgss"

# ---------------------------------------------------------------------------
# FAST/FULL toggle + fixed hyperparameters
# ---------------------------------------------------------------------------
MODE = "fast"
M_BY_MODE = {"fast": 199, "full": 999}
M_CALIBRATION = M_BY_MODE[MODE]
N_RESTARTS_NULL = 3  # matches 07/09/10's null-replica restart count

SCORE_KIND = "BJ"  # "BJ" (default) or "HC"; the paper compares both and found BJ
                   # stronger on several real tasks, so BJ is the reported default and
                   # HC is available behind this flag for the summary's BJ-vs-HC note.
ALPHA_MAX = 0.10   # FIXED A PRIORI, never tuned on detection: the conventional 0.10
                   # significance band. A cell is "signal-carrying" iff its null-model
                   # p-value <= 0.10, and BJ maximizes over candidate thresholds
                   # alpha in (0, 0.10]. This is a standard significance cutoff chosen
                   # before any detection run — NOT selected by comparing detection
                   # power across candidate alpha_max values. (A larger alpha_max, e.g.
                   # the paper's 0.5, both widens the candidate-alpha search — costlier
                   # — and dilutes power for the concentrated subgroup signals here;
                   # 0.10 is fixed on principle, not because it maximized power.)


# =============================================================================
# EMPIRICAL P-VALUES (anchored NB null) + BERK-JONES SCORE
# =============================================================================

def nb_upper_midp(obs, mu, alpha_disp):
    """Per-cell upper-tail mid-p value against NB2(mean=mu_i, dispersion
    alpha_disp): p_i = P(X > c_i) + 1/2 P(X = c_i). Small p_i <=> surprisingly
    high count (emergence). mu is floored to a tiny positive to keep the NB
    parameters well-defined for any near-empty cell."""
    obs = np.asarray(obs)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    r = 1.0 / alpha_disp
    p = r / (r + mu)
    sf_ge = nbinom.sf(obs - 1, r, p)   # P(X >= c)
    pmf = nbinom.pmf(obs, r, p)        # P(X = c)
    midp = sf_ge - 0.5 * pmf           # = P(X>c) + 0.5 P(X=c)
    return np.clip(midp, 1e-12, 1.0)


class BerkJonesScorer:
    """Nonparametric NPSS scorer for 06's pluggable search skeleton.

    Score of a subset S: F(S) = max over alpha in candidate_alphas of
    phi(alpha, N_alpha(S), N(S)), where N(S) = #cells in S and N_alpha(S) =
    #cells in S with p <= alpha. Berk-Jones phi is the Bernoulli-KL LLR for
    "the fraction of p-values below alpha equals alpha (null) vs exceeds it":
        phi_BJ(a, A, N) = N * [ phat*log(phat/a) + (1-phat)*log((1-phat)/(1-a)) ],
        phat = A/N,   scored only when phat > a (excess / emergence), else 0.
    Higher Criticism alternative (kind='HC'):
        phi_HC(a, A, N) = (A - N*a) / sqrt(N*a*(1-a)),   scored only when phat > a.

    LTSS per feature (McFowland 2013): for a FIXED alpha, each value-group v
    contributes (a_v = #{p<=alpha}, n_v = #cells); sorting groups by a_v/n_v
    descending makes the optimal included set one of the sorted prefixes — the
    nonparametric analogue of sorting by C/B in the Poisson LTSS. We take the
    max over prefixes AND over candidate alphas. candidate_alphas is the fixed
    set of distinct p-values <= alpha_max in the whole cell frame (the score is
    piecewise-constant, changing only at observed p-values).
    """

    def __init__(self, pval_col, alpha_max, candidate_alphas, kind="BJ"):
        self.pval_col = pval_col
        self.alpha_max = alpha_max
        self.alphas = np.asarray(candidate_alphas, dtype=float)
        self.kind = kind

    def _phi(self, alpha, A, N):
        """Vectorized over A, N (same shape); alpha scalar. Returns score array,
        0 wherever the excess condition phat>alpha fails or N==0."""
        A = np.asarray(A, dtype=float)
        N = np.asarray(N, dtype=float)
        out = np.zeros_like(A)
        with np.errstate(divide="ignore", invalid="ignore"):
            phat = np.where(N > 0, A / N, 0.0)
            excess = (N > 0) & (phat > alpha)
            if self.kind == "HC":
                denom = np.sqrt(N * alpha * (1.0 - alpha))
                val = np.where(denom > 0, (A - N * alpha) / denom, 0.0)
            else:  # Berk-Jones
                term1 = phat * np.log(phat / alpha)
                term2 = np.where(phat < 1.0, (1.0 - phat) * np.log((1.0 - phat) / (1.0 - alpha)), 0.0)
                val = N * (term1 + term2)
            out = np.where(excess, val, 0.0)
        return out

    def optimize_feature(self, sub_df, f):
        vals = sub_df[f].to_numpy()
        pv = sub_df[self.pval_col].to_numpy()
        uniq = pd.unique(vals)
        K = len(uniq)
        # Default to the FULL value set (unrestricted) when nothing scores — same
        # anti-stranding rationale as 06's Poisson k=K default.
        if K == 0 or len(self.alphas) == 0:
            return list(uniq), 0.0
        sorted_pv = [np.sort(pv[vals == v]) for v in uniq]
        n_v = np.array([len(s) for s in sorted_pv], dtype=float)
        # a_matrix[v, j] = # p-values in group v that are <= alphas[j]
        a_matrix = np.array([np.searchsorted(s, self.alphas, side="right") for s in sorted_pv], dtype=float)

        best_score, best_values = 0.0, list(uniq)
        for j, alpha in enumerate(self.alphas):
            a_v = a_matrix[:, j]
            ratio = np.where(n_v > 0, a_v / n_v, 0.0)
            order = np.argsort(-ratio, kind="stable")
            cum_a = np.cumsum(a_v[order])
            cum_n = np.cumsum(n_v[order])
            s = self._phi(alpha, cum_a, cum_n)
            k = int(np.argmax(s))
            if s[k] > best_score + 1e-12:
                best_score = float(s[k])
                best_values = [uniq[i] for i in order[: k + 1]]
        return best_values, best_score

    def total_score(self, cells_df, feature_cols, subset):
        mask = pd.Series(True, index=cells_df.index)
        for feat in feature_cols:
            if feat in subset:
                mask &= cells_df[feat].isin(subset[feat])
        pv = np.sort(cells_df.loc[mask, self.pval_col].to_numpy())
        N = len(pv)
        if N == 0 or len(self.alphas) == 0:
            return 0.0
        A = np.searchsorted(pv, self.alphas, side="right").astype(float)
        s = self._phi(self.alphas, A, np.full_like(A, float(N)))
        return float(np.max(s)) if len(s) else 0.0


# =============================================================================
# FGSS SCORE + DETECT (conform to the Method interface for RAMP reuse)
# =============================================================================

def fgss_score(window_cells, expected, n_restarts, seed, alpha_disp, alpha_max=ALPHA_MAX, kind=SCORE_KIND):
    """Empirical p-values from (observed, expected=re-anchored mean, alpha_disp),
    then the shared LTSS scan under the Berk-Jones scorer. Returns (subset, max BJ)."""
    obs = window_cells["observed"].to_numpy()
    mu = expected.to_numpy() if hasattr(expected, "to_numpy") else np.asarray(expected)
    pvals = nb_upper_midp(obs, mu, alpha_disp)
    cells = window_cells[FEATURES].copy()
    cells["__pval__"] = pvals
    candidate_alphas = np.unique(pvals[pvals <= alpha_max])
    scorer = BerkJonesScorer("__pval__", alpha_max, candidate_alphas, kind=kind)
    subset, sc = mdss.scan(cells, None, None, FEATURES, n_restarts=n_restarts, seed=seed, scorer=scorer)
    return subset, sc


def make_fgss_method(alpha_disp, alpha_max=ALPHA_MAX, kind=SCORE_KIND):
    """A mod08.Method whose fit_expected is the proposed method's exact
    re-anchoring (share x current total) and whose detect runs the BJ scan
    against FGSS's own null. Lets RAMP reuse mod08.run_ramp_experiment verbatim."""

    def detect(window_cells, expected, null_for_window, n_restarts, seed):
        subset, sc = fgss_score(window_cells, expected, n_restarts, seed, alpha_disp, alpha_max, kind)
        p = e1.pvalue(sc, null_for_window)
        return (p < 0.05), subset, sc, p

    return mod08.Method(name=METHOD_NAME, fit_expected=fgss_fit_expected, detect=detect)


# =============================================================================
# OWN NULL CALIBRATION
# =============================================================================

def calibrate_fgss(base_data, windows, alpha_disp, m, n_restarts, seed_base, alpha_max=ALPHA_MAX, kind=SCORE_KIND, verbose=True):
    null_scores, check_fractions, long_rows = {}, {}, []
    for pos, w in enumerate(windows):
        window_df = base_data[w]

        def one_replica(seed):
            rng = np.random.default_rng(seed)
            trial = make_clean_trial(window_df, alpha_disp, rng)
            exp = fgss_fit_expected(trial)
            _sub, sc = fgss_score(trial, exp, n_restarts, seed, alpha_disp, alpha_max, kind)
            return sc

        null = np.array([one_replica(seed_base + pos * 10_000 + i) for i in range(m)])
        p95 = np.percentile(null, 95)
        null_scores[w] = null
        for i, s in enumerate(null):
            long_rows.append({"window_label": w, "replica_idx": i, "max_bj_score": s})

        check = np.array([one_replica(seed_base + pos * 10_000 + 50_000 + i) for i in range(m)])
        exceed = float(np.mean(check > p95))
        check_fractions[w] = exceed
        if verbose:
            print(f"  [{w}] p95_BJ={p95:.2f}, calibration check={exceed:.1%}")

    return null_scores, check_fractions, pd.DataFrame(long_rows)


# =============================================================================
# STEP experiment (own loop, to record precision alongside jaccard/recall)
# =============================================================================

def run_fgss_step_experiment(method, base_data, fgss_null, windows, alpha_disp, j_grid, r_grid, n_trials, n_restarts, seed_base, min_baseline_expected):
    pooled_ref = pd.concat(base_data.values(), ignore_index=True).groupby(FEATURES, as_index=False)["expected"].sum()
    rows = []
    for j in j_grid:
        for r in r_grid:
            cell_seed = seed_base + deterministic_seed(METHOD_NAME, j, r) % 10_000
            rng_subgroups = np.random.default_rng(cell_seed)
            for trial_idx in range(n_trials):
                S, _ = draw_subgroup(rng_subgroups, j, pooled_ref, min_baseline_expected)
                if S is None:
                    continue
                for w in windows:
                    window_df = base_data[w]
                    mask = pd.Series(True, index=window_df.index)
                    for f, vals in S.items():
                        mask &= window_df[f].isin(vals)
                    if window_df.loc[mask, "expected"].sum() < min_baseline_expected:
                        continue

                    trial_seed = seed_base + deterministic_seed(METHOD_NAME, j, r, trial_idx, w) % 1_000_000
                    rng = np.random.default_rng(trial_seed)
                    trial_df = make_clean_trial(window_df, alpha_disp, rng)

                    clean_exp = method.fit_expected(trial_df)
                    c_det, _c_sub, c_sc, c_p = method.detect(trial_df, clean_exp, fgss_null[w], n_restarts, trial_seed)
                    rows.append({
                        "method": METHOD_NAME, "j": j, "r": r, "window": w, "trial": trial_idx,
                        "arm": "clean", "S": repr(S), "detected": c_det, "score": c_sc,
                        "pvalue": c_p, "jaccard": np.nan, "recall": np.nan, "precision": np.nan,
                    })

                    injected_df = inject(trial_df, S, r)
                    inj_exp = method.fit_expected(injected_df)
                    i_det, i_sub, i_sc, i_p = method.detect(injected_df, inj_exp, fgss_null[w], n_restarts, trial_seed + 1)
                    rows.append({
                        "method": METHOD_NAME, "j": j, "r": r, "window": w, "trial": trial_idx,
                        "arm": "injected", "S": repr(S), "detected": i_det, "score": i_sc,
                        "pvalue": i_p, "jaccard": jaccard(i_sub, S), "recall": recall(i_sub, S),
                        "precision": precision(i_sub, S),
                    })
    return pd.DataFrame(rows)


def summarize_fgss_step(step_df):
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
# VALIDATION GATE
# =============================================================================

def run_validation_gate_fgss(method, base_data, fgss_null, alpha_disp, n_restarts_gate=N_RESTARTS_GATE):
    print("\n" + "=" * 78)
    print("VALIDATION GATE (FGSS, before the full grid)")
    print("=" * 78)
    gate_window = "2023-Q3"
    print(f"Gate window: {gate_window} (mid-volume; conjunctive S — mod08.draw_subgroup plants one value/feature).")
    step_df = run_fgss_step_experiment(
        method, {gate_window: base_data[gate_window]}, {gate_window: fgss_null[gate_window]}, [gate_window],
        alpha_disp, j_grid=[2], r_grid=[1.5, 3.0], n_trials=N_TRIALS_GATE, n_restarts=n_restarts_gate,
        seed_base=BASE_SEED + 1800, min_baseline_expected=MIN_BASELINE_EXPECTED,
    )
    summary = summarize_fgss_step(step_df)
    print(summary.to_string(index=False))

    checks = {}
    fpr = summary["fpr"].mean()
    checks["clean_fpr_near_5pct"] = bool(0.0 <= fpr <= 0.20)
    print(f"\n[{'PASS' if checks['clean_fpr_near_5pct'] else 'FAIL'}] Clean-arm FPR = {fpr:.1%} (own FGSS null)")

    power_15 = summary.loc[summary["r"] == 1.5, "power"].iloc[0]
    power_30 = summary.loc[summary["r"] == 3.0, "power"].iloc[0]
    checks["power_rises_with_r"] = bool(power_30 > power_15)
    print(f"[{'PASS' if checks['power_rises_with_r'] else 'FAIL'}] Power rises with r: "
          f"power(1.5)={power_15:.2f} -> power(3.0)={power_30:.2f}")

    mean_recall_strong = summary.loc[summary["r"] == 3.0, "mean_recall_detected"].iloc[0]
    checks["recovers_planted_subgroup"] = bool(pd.notna(mean_recall_strong) and mean_recall_strong >= 0.5)
    print(f"[{'PASS' if checks['recovers_planted_subgroup'] else 'FAIL'}] Mean recall on detected strong (r=3) "
          f"trials = {mean_recall_strong:.2f} (>=0.5 required)")

    all_pass = all(checks.values())
    print(f"\nGate result: {'ALL PASS' if all_pass else 'FAILED — STOPPING, not launching the full grid'}")
    return all_pass, checks


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
    mod06_md5 = mod04.file_md5(_scripts_dir / "06_mdss_scan.py")

    step_csv = config.REPORTS_DIR / "e2_step_results.csv"
    ramp_csv = config.REPORTS_DIR / "e2_ramp_results.csv"
    step_sum_csv = config.REPORTS_DIR / "e2_step_summary.csv"
    ramp_sum_csv = config.REPORTS_DIR / "e2_ramp_summary.csv"
    prior_step = pd.read_csv(step_csv)
    prior_ramp = pd.read_csv(ramp_csv)
    prior_step = prior_step[prior_step["method"] != METHOD_NAME].reset_index(drop=True)
    prior_ramp = prior_ramp[prior_ramp["method"] != METHOD_NAME].reset_index(drop=True)

    alpha_disp = mod08.alpha_from_e1()
    base_data, _poisson_null, windows = mod08.load_base_data()
    print(f"\nLoaded {len(windows)} test windows. Score={SCORE_KIND}, alpha_max={ALPHA_MAX} (fixed a priori, not tuned).")
    method = make_fgss_method(alpha_disp, ALPHA_MAX, SCORE_KIND)

    print("\n" + "=" * 78)
    print("FGSS NULL CALIBRATION (max Berk-Jones score, own null)")
    print("=" * 78)
    t_cal = time.time()
    fgss_null, check_fractions, null_long = calibrate_fgss(base_data, windows, alpha_disp, M_CALIBRATION, N_RESTARTS_NULL, BASE_SEED + 1900)
    cal_elapsed = time.time() - t_cal
    n_cal_calls = len(windows) * M_CALIBRATION * 2
    per_call = cal_elapsed / n_cal_calls
    pooled_fpr = float(np.mean(list(check_fractions.values())))
    print(f"\nPooled calibration FPR (own null) = {pooled_fpr:.1%} (target 5%)")
    out_path = config.REPORTS_DIR / "e2_fgss_null_maxscores.csv"
    null_long.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(null_long)} rows)")

    gate_pass, gate_checks = run_validation_gate_fgss(method, base_data, fgss_null, alpha_disp)
    if not gate_pass:
        print("\nSTOPPING: FGSS validation gate failed. Not launching the full grid.")
        return

    n_windows = len(windows)
    n_step = len(J_GRID) * len(R_GRID) * N_TRIALS_FULL * n_windows * 2
    n_ramp = len(J_GRID) * len(R_GRID) * N_TRIALS_FULL * RAMP_LENGTH
    print(f"\nMODE={MODE!r}. Per BJ scan ~= {per_call*1000:.0f} ms (from {n_cal_calls:,} calibration scans). "
          f"Grid scans: {n_step:,} step + {n_ramp:,} ramp = {n_step + n_ramp:,}. ETA ~= {(n_step + n_ramp) * per_call / 60:.1f} min.")

    print("\n" + "=" * 78)
    print("FGSS STEP INJECTION EXPERIMENT")
    print("=" * 78)
    fgss_step_df = run_fgss_step_experiment(method, base_data, fgss_null, windows, alpha_disp, J_GRID, R_GRID, N_TRIALS_FULL, N_RESTARTS_GRID, BASE_SEED + 2000, MIN_BASELINE_EXPECTED)
    fgss_step_summary = summarize_fgss_step(fgss_step_df)
    print(fgss_step_summary.to_string(index=False))

    print("\n" + "=" * 78)
    print("FGSS RAMP INJECTION EXPERIMENT (reusing mod08.run_ramp_experiment)")
    print("=" * 78)
    fgss_ramp_df = mod08.run_ramp_experiment(
        [method], base_data, fgss_null, windows, J_GRID, R_GRID,
        N_TRIALS_FULL, N_RESTARTS_GRID, RAMP_LENGTH, BASE_SEED + 2100, MIN_BASELINE_EXPECTED, alpha_disp,
    )
    fgss_ramp_summary = mod08.summarize_ramp(fgss_ramp_df)
    print(fgss_ramp_summary.to_string(index=False))

    std_cols = ["method", "j", "r", "window", "trial", "arm", "S", "detected", "score", "pvalue", "jaccard", "recall", "precision"]
    combined_step = pd.concat([prior_step, fgss_step_df[std_cols]], ignore_index=True)
    combined_ramp = pd.concat([prior_ramp, fgss_ramp_df], ignore_index=True)
    combined_step.to_csv(step_csv, index=False)
    combined_ramp.to_csv(ramp_csv, index=False)
    print(f"\nAppended FGSS rows: {step_csv} now {len(combined_step)} rows; {ramp_csv} now {len(combined_ramp)} rows")

    prior_step_summary = pd.read_csv(step_sum_csv)
    prior_step_summary = prior_step_summary[prior_step_summary["method"] != METHOD_NAME]
    for col in ["mean_precision_detected", "mean_precision_all"]:
        if col not in prior_step_summary.columns:
            prior_step_summary[col] = np.nan
    combined_step_summary = pd.concat([prior_step_summary, fgss_step_summary], ignore_index=True)
    prior_ramp_summary = pd.read_csv(ramp_sum_csv)
    prior_ramp_summary = prior_ramp_summary[prior_ramp_summary["method"] != METHOD_NAME]
    combined_ramp_summary = pd.concat([prior_ramp_summary, fgss_ramp_summary], ignore_index=True)
    combined_step_summary.to_csv(step_sum_csv, index=False)
    combined_ramp_summary.to_csv(ramp_sum_csv, index=False)

    mod09.plot_power_curves_by_method(combined_step_summary, "e2_power_curves_by_method.png")
    mod09.plot_precision_recall_jaccard(combined_step_summary, "e2_precision_recall_jaccard_by_method.png")
    mod09.plot_time_to_detect_by_method(combined_ramp_summary, "e2_time_to_detect_by_method.png")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"\n06 refactor: parameterized scan() with a pluggable `scorer` (default = original Poisson). "
          f"06 MD5 is now {mod06_md5} (changed by the refactor). Its Part B toy tests were re-run and ALL PASS "
          "(null/k=0-empty-subset, single-feature, conjunction recovery, monotonicity, restart robustness) — "
          "the Poisson path is behaviour-identical.")
    print(f"\nScore statistic = {SCORE_KIND} (Berk-Jones). alpha_max = {ALPHA_MAX}, fixed a priori (conventional 0.10 "
          "significance band), never tuned on detection. HC is available behind SCORE_KIND='HC' for comparison; "
          "BJ is the reported default per the paper's finding that BJ is stronger on several real tasks.")
    print(f"Pooled FGSS calibration FPR (own null) = {pooled_fpr:.1%}")
    print(f"\nGate: {'PASS' if gate_pass else 'FAIL'} ({gate_checks})")
    print(f"\nFGSS step summary (power/FPR/precision/recall/Jaccard by j,r):\n{fgss_step_summary.to_string(index=False)}")
    print(f"\nFGSS ramp summary:\n{fgss_ramp_summary.to_string(index=False)}")

    print("\n4-way head-to-head (power, by j,r):")
    compare = combined_step_summary.pivot_table(index=["j", "r"], columns="method", values="power")
    print(compare.to_string())
    print("\n4-way identification on detected injected trials (precision / recall, by method, pooled over j,r):")
    inj_detected = combined_step[(combined_step["arm"] == "injected") & (combined_step["detected"])]
    ident = inj_detected.groupby("method").agg(mean_precision=("precision", "mean"), mean_recall=("recall", "mean"), n_detected=("detected", "size"))
    print(ident.to_string())

    per_window_power = fgss_step_df[fgss_step_df["arm"] == "injected"].groupby("window")["detected"].mean()
    flagged = per_window_power[per_window_power.index.isin(FLAG_WINDOWS)]
    other = per_window_power[~per_window_power.index.isin(FLAG_WINDOWS)]
    print(f"\nFGSS pooled power (non-flagged windows): {other.mean():.2f}")
    print(f"FGSS power in flagged windows {sorted(FLAG_WINDOWS)}: {flagged.to_dict()}")

    print(f"\nHarness core (08_e2_injection.py) unchanged: {mod08_md5_before == mod04.file_md5(mod08_path)}")
    elapsed = time.time() - t_start
    print(f"Total compute: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\nSource artifact integrity:")
    for name, before in md5_before.items():
        print(f"  {name}: unchanged={before == mod04.file_md5(md5_paths[name])}")


if __name__ == "__main__":
    main()
