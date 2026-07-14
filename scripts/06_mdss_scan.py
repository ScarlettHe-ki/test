"""Expectation-based Poisson MDSS (Multivariate Bayesian/Deterministic Subset
Scan) implemented from scratch, per Neill (2012) "Fast subset scan for spatial
pattern detection" (LTSS) and the expectation-based Poisson scan statistic
(Neill 2009, "Expectation-based scan statistics for monitoring spatial
time series data").

This module is IMPORTABLE: score(), cells_in_subset(), and scan() are the
public API that later scripts (E1 calibration, E2 injection, E3 discovery)
call directly. All validation and the real-data smoke test live inside
main(), guarded by `if __name__ == "__main__"`, so importing this module has
no side effects.

Read-only on config.ACTIVE_DATA_PATH, config.PROCESSED_DATA_PATH, and
config.ANCHORED_EXPECTED_COUNTS_PATH (writes no data files).
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

FEATURES = mod04.FEATURES
REDUCED_CONFIDENCE_WINDOWS = {"2022-Q3", "2022-Q4"}


# =============================================================================
# PART A — the scan statistic (public API)
# =============================================================================

def score(C, B):
    """Expectation-based Poisson scan statistic for a single candidate subset.

    Derivation: under the null, counts in the subset are Poisson(B) (B = sum
    of expected counts). Under the alternative, counts are Poisson(q*B) for
    an unknown multiplicative risk q > 1 (elevated rate only — q <= 1, i.e. a
    DEFICIT, is not scored; F=0 there, since we're scanning for emergence).
    The log-likelihood ratio is:
        LLR(q) = C*log(q) - B*(q-1)
    Maximising over q > 1 (d/dq = C/q - B = 0) gives q_hat = C/B, valid
    whenever C > B (else the unconstrained maximiser q_hat <= 1 is outside
    the q > 1 domain, so the constrained max is at q=1, giving LLR=0).
    Substituting q_hat back in:
        F(C,B) = C*log(C/B) - B*(C/B - 1) = C*log(C/B) - (C - B),  for C > B
        F(C,B) = 0,                                                 otherwise

    This is the ONLY place the statistic is defined; every other function in
    this module calls score() rather than re-deriving it.
    """
    if B <= 0 or C <= B:
        return 0.0
    return C * np.log(C / B) - (C - B)


def _subset_mask(cells_df, feature_cols, subset):
    mask = pd.Series(True, index=cells_df.index)
    for f in feature_cols:
        if f in subset:
            mask &= cells_df[f].isin(subset[f])
    return mask


def cells_in_subset(cells_df, feature_cols, subset):
    """Return the rows of cells_df belonging to `subset` (a feature -> list-of-
    included-values dict; features absent from the dict are unrestricted)."""
    return cells_df.loc[_subset_mask(cells_df, feature_cols, subset)]


def _total_score(cells_df, obs_col, exp_col, feature_cols, subset):
    mask = _subset_mask(cells_df, feature_cols, subset)
    return score(cells_df.loc[mask, obs_col].sum(), cells_df.loc[mask, exp_col].sum())


def _ltss_optimal_subset(agg_df, value_col, c_col="C", b_col="B"):
    """Linear-time subset scanning (LTSS), Neill (2012).

    WHY sorting + prefix search is exact, not a heuristic: for any FIXED risk
    multiplier q, the log-likelihood-ratio contribution of including group i
    (with aggregated observed/expected C_i, B_i) is linear in (C_i, B_i):
        f_i(q) = C_i*log(q) - B_i*(q-1)
    which is positive iff C_i/B_i > (q-1)/log(q) — i.e. for any fixed q, the
    LLR-maximising set of groups to include is exactly {i : C_i/B_i > some
    threshold(q)}, which is a "prefix" once groups are sorted by C_i/B_i
    descending. The true scan statistic jointly maximises over BOTH the
    subset S and its own q_hat = C(S)/B(S); since q_hat is itself just some
    value of q, the globally optimal S must be the threshold-optimal set for
    q = q_hat, and is therefore guaranteed to be ONE of the K sorted prefixes.
    So instead of enumerating all 2^K subsets of this feature's values, we
    only need to check the K prefixes after an O(K log K) sort — hence
    "linear-time" (per feature, given the sort).
    """
    d = agg_df[agg_df[b_col] > 0].copy()
    d["ratio"] = d[c_col] / d[b_col]
    d = d.sort_values("ratio", ascending=False).reset_index(drop=True)
    K = len(d)
    if K == 0:
        return [], 0.0

    cum_c = d[c_col].cumsum().to_numpy()
    cum_b = d[b_col].cumsum().to_numpy()

    # Default to the FULL prefix (k=K, i.e. "unrestricted") rather than the
    # empty prefix (k=0) when no strictly-positive-scoring prefix exists.
    # This matters mechanically: if this feature's chosen subset were empty,
    # it would zero out the AND-mask for every other feature on the next
    # sweep, permanently stranding them at their stale initial values instead
    # of letting them re-optimize against the other features' real subsets.
    best_k = K
    best_val = score(cum_c[K - 1], cum_b[K - 1])
    for k in range(1, K):
        s = score(cum_c[k - 1], cum_b[k - 1])
        if s > best_val + 1e-12:
            best_val = s
            best_k = k
    return d[value_col].iloc[:best_k].tolist(), best_val


class _PoissonScorer:
    """Default scorer: the expectation-based Poisson statistic exactly as this
    module has always computed it. Extracted verbatim (groupby -> C,B ->
    _ltss_optimal_subset; full-subset score via _total_score) so that the
    coordinate-ascent/multi-restart SEARCH SKELETON below can be reused by
    other scan statistics that also satisfy the LTSS property (e.g. the
    nonparametric Berk-Jones scan in 11_fgss.py) WITHOUT copy-pasting the
    ascent. Passing scorer=None to scan() reconstructs this object, so the
    Poisson path is behaviour-identical to before the refactor."""

    def __init__(self, obs_col, exp_col):
        self.obs_col = obs_col
        self.exp_col = exp_col

    def optimize_feature(self, sub_df, f):
        agg = (
            sub_df.groupby(f, as_index=False)[[self.obs_col, self.exp_col]]
            .sum()
            .rename(columns={self.obs_col: "C", self.exp_col: "B"})
        )
        return _ltss_optimal_subset(agg, f)

    def total_score(self, cells_df, feature_cols, subset):
        return _total_score(cells_df, self.obs_col, self.exp_col, feature_cols, subset)


def _random_init(cells_df, feature_cols, rng):
    subset = {}
    for f in feature_cols:
        vals = cells_df[f].unique()
        while True:
            incl = vals[rng.random(len(vals)) < 0.5]
            if len(incl) > 0:
                break
        subset[f] = list(incl)
    return subset


def _coordinate_ascent(cells_df, feature_cols, rng, scorer, max_sweeps=50):
    """One run of the MDSS ascent from a random initial assignment.

    Alternates over the 8 features: each step fixes all OTHER features'
    current value-subsets, restricts cells_df to rows matching them, and
    replaces the free feature's subset with its LTSS-optimal choice given
    that restriction (computed by `scorer.optimize_feature`). Repeats sweeps
    until no feature's subset changes (converged) or max_sweeps is hit (safety
    cap; the score is non-decreasing every step, so this is a coordinate-ascent
    local optimum, not necessarily the global one — see scan()'s multi-restart).

    The ascent itself is score-agnostic: `scorer` supplies both the per-feature
    LTSS optimizer and the full-subset score, so the identical skeleton drives
    the Poisson scan and the nonparametric Berk-Jones (FGSS) scan.
    """
    subset = _random_init(cells_df, feature_cols, rng)

    for _ in range(max_sweeps):
        changed = False
        for f in feature_cols:
            other_mask = pd.Series(True, index=cells_df.index)
            for f2 in feature_cols:
                if f2 != f and f2 in subset:
                    other_mask &= cells_df[f2].isin(subset[f2])
            sub_df = cells_df.loc[other_mask]
            if sub_df.empty:
                continue
            best_values, _ = scorer.optimize_feature(sub_df, f)
            if set(best_values) != set(subset[f]):
                changed = True
            subset[f] = best_values
        if not changed:
            break

    # Cleanup: a feature whose optimal subset is ALL of its distinct values
    # (or ended up empty, degenerate) is dropped from the returned dict, per
    # the "absent from the dict = unrestricted" convention.
    final_score = scorer.total_score(cells_df, feature_cols, subset)
    clean_subset = {}
    for f in feature_cols:
        all_vals = set(cells_df[f].unique())
        if subset[f] and set(subset[f]) != all_vals:
            clean_subset[f] = sorted(subset[f], key=str)
    return clean_subset, final_score


def scan(cells_df, obs_col, exp_col, feature_cols, n_restarts=20, seed=None, return_all_restarts=False, scorer=None):
    """Multi-restart MDSS scan: the public entry point.

    The coordinate ascent (_coordinate_ascent) is a LOCAL optimiser — each
    step is exact (LTSS) given the other features' current values, but the
    joint alternating procedure is not guaranteed to reach the GLOBAL
    optimum over all 8 features at once. Running from R random restarts and
    keeping the best-scoring result approximates the global optimum; it does
    not guarantee it, but empirically (see Part B, Test 5) a modest R finds
    the true best subset with high probability even when individual restarts
    get trapped by a weaker, decoy local optimum.

    `scorer` plugs in the scan statistic. When None (the default, used by every
    existing caller), it is the expectation-based Poisson statistic built from
    obs_col/exp_col — behaviour-identical to before this parameter existed. A
    caller may instead pass any object exposing optimize_feature(sub_df, f) ->
    (values, score) and total_score(cells_df, feature_cols, subset) -> float
    (e.g. 11_fgss.py's Berk-Jones scorer, which ignores obs_col/exp_col and
    reads a precomputed per-cell p-value column).

    Returns (best_subset, best_score) by default; pass return_all_restarts=True
    to also get the list of every restart's (subset, score) for diagnostics.
    """
    if scorer is None:
        scorer = _PoissonScorer(obs_col, exp_col)
    rng = np.random.default_rng(seed)
    best_subset, best_score_val = {}, -np.inf
    all_results = []
    for _ in range(n_restarts):
        sub, sc = _coordinate_ascent(cells_df, feature_cols, rng, scorer)
        all_results.append((sub, sc))
        if sc > best_score_val:
            best_subset, best_score_val = sub, sc
    if return_all_restarts:
        return best_subset, best_score_val, all_results
    return best_subset, best_score_val


# =============================================================================
# PART B — validation on toy data with planted signals
# =============================================================================

def _toy_2feature_table(peril_vals, region_vals, inflate_fn, base_exp=10, base_obs=10):
    rows = []
    for p in peril_vals:
        for r in region_vals:
            rows.append({"peril": p, "region": r, "obs": inflate_fn(p, r), "exp": base_exp})
    return pd.DataFrame(rows)


def validate_null_sanity():
    peril, region = list("ABCD"), list("WXYZ")
    df = _toy_2feature_table(peril, region, lambda p, r: 10)
    subset, sc = scan(df, "obs", "exp", ["peril", "region"], n_restarts=10, seed=1)
    ok = abs(sc) < 1e-6 and subset == {}
    print(f"[{'PASS' if ok else 'FAIL'}] Test 1 (null sanity): score={sc:.6f}, subset={subset}")
    return ok


def validate_single_feature_signal():
    peril, region = list("ABCD"), list("WXYZ")
    df = _toy_2feature_table(peril, region, lambda p, r: 20 if p in ("A", "B") else 10)
    subset, sc = scan(df, "obs", "exp", ["peril", "region"], n_restarts=10, seed=2)
    matched = set(subset.keys()) == {"peril"} and set(subset.get("peril", [])) == {"A", "B"}
    if matched:
        sub_df = cells_in_subset(df, ["peril", "region"], subset)
        q_hat = sub_df["obs"].sum() / sub_df["exp"].sum()
    else:
        q_hat = float("nan")
    ok = matched and abs(q_hat - 2.0) < 1e-6
    print(f"[{'PASS' if ok else 'FAIL'}] Test 2 (single-feature signal): subset={subset}, "
          f"q_hat={q_hat:.4f} (planted x2)")
    return ok


def validate_conjunction_signal():
    peril, region = list("ABCD"), list("WXYZ")
    df = _toy_2feature_table(peril, region, lambda p, r: 30 if (p in ("A", "B") and r == "W") else 10)
    subset, sc = scan(df, "obs", "exp", ["peril", "region"], n_restarts=20, seed=3)
    matched = (
        set(subset.keys()) == {"peril", "region"}
        and set(subset.get("peril", [])) == {"A", "B"}
        and set(subset.get("region", [])) == {"W"}
    )
    if matched:
        sub_df = cells_in_subset(df, ["peril", "region"], subset)
        q_hat = sub_df["obs"].sum() / sub_df["exp"].sum()
    else:
        q_hat = float("nan")
    ok = matched and abs(q_hat - 3.0) < 1e-6
    print(f"[{'PASS' if ok else 'FAIL'}] Test 3 (conjunction signal, AND-across features): "
          f"subset={subset}, q_hat={q_hat:.4f} (planted x3, intersection only)")
    return ok


def validate_score_monotonicity():
    B = 100.0
    multipliers = [1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    scores = [score(B * q, B) for q in multipliers]
    ok = all(scores[i] < scores[i + 1] for i in range(len(scores) - 1))
    print(f"[{'PASS' if ok else 'FAIL'}] Test 4 (score monotonicity): "
          f"{list(zip(multipliers, [round(s, 2) for s in scores]))}")
    return ok


def validate_restart_robustness(n_restarts=20):
    peril6, region5, risk3 = list("ABCDEF"), list("VWXYZ"), list("PQR")
    rows = []
    for p in peril6:
        for r in region5:
            for k in risk3:
                obs = 10
                if p in ("A", "B") and r == "V" and k == "P":
                    obs = 60  # true global-best signal, x6
                elif p in ("C", "D") and r == "W" and k == "Q":
                    obs = 40  # weaker decoy signal, x4 — designed to trap some restarts
                rows.append({"peril": p, "region": r, "risk": k, "obs": obs, "exp": 10})
    df = pd.DataFrame(rows)
    true_subset = {"peril": ["A", "B"], "region": ["V"], "risk": ["P"]}

    best_subset, best_score_val, all_results = scan(
        df, "obs", "exp", ["peril", "region", "risk"], n_restarts=n_restarts, seed=5, return_all_restarts=True
    )
    hits = sum(
        1 for sub, _ in all_results
        if set(sub.get("peril", [])) == {"A", "B"} and set(sub.get("region", [])) == {"V"} and set(sub.get("risk", [])) == {"P"}
    )
    fraction = hits / n_restarts
    overall_matched = (
        set(best_subset.get("peril", [])) == {"A", "B"}
        and set(best_subset.get("region", [])) == {"V"}
        and set(best_subset.get("risk", [])) == {"P"}
    )
    print(
        f"[{'PASS' if overall_matched else 'FAIL'}] Test 5 (restart robustness): "
        f"a weaker decoy signal (x4) was planted alongside the true signal (x6) to induce local optima. "
        f"Per-restart recovery of the true global optimum: {hits}/{n_restarts} ({fraction:.0%}). "
        f"Multi-restart scan() best result: {'matches true subset' if overall_matched else 'DID NOT match'}."
    )
    return overall_matched, fraction


def run_validation():
    print("=" * 78)
    print("PART B — VALIDATION ON TOY DATA WITH PLANTED SIGNALS")
    print("=" * 78)
    results = {}
    results["null_sanity"] = validate_null_sanity()
    results["single_feature"] = validate_single_feature_signal()
    results["conjunction"] = validate_conjunction_signal()
    results["monotonicity"] = validate_score_monotonicity()
    restart_ok, restart_fraction = validate_restart_robustness()
    results["restart_robustness"] = restart_ok

    all_pass = all(results.values())
    print(f"\nAll Part B tests passed: {all_pass}")
    if not all_pass:
        failed = [k for k, v in results.items() if not v]
        print(f"FAILED: {failed}. STOPPING — not proceeding to Part C.")
    return all_pass, results, restart_fraction


# =============================================================================
# PART C — single real scan (smoke test)
# =============================================================================

def run_real_scan():
    print("\n" + "=" * 78)
    print("PART C — SINGLE REAL SCAN (smoke test, not the experiment)")
    print("=" * 78)

    df = pd.read_csv(config.ANCHORED_EXPECTED_COUNTS_PATH)
    print(f"\nLoaded {len(df)} (cell x window) rows from {config.ANCHORED_EXPECTED_COUNTS_PATH.name}")

    pooled = df.groupby(FEATURES, as_index=False)[["observed", "expected"]].sum()
    print(
        f"Pooled across all {df['window_idx'].nunique()} test windows (2022-Q1 .. 2024-Q4) into "
        f"{len(pooled)} unique cells. Rationale: this is a smoke test of correctness on real data, "
        "not the discovery analysis — pooling maximises power/robustness for a first sanity check. "
        "A proper per-window emerging-driver scan is exactly what script E3 will do."
    )
    print(f"Pooled total observed={pooled['observed'].sum()}, total expected={pooled['expected'].sum():.1f}")

    best_subset, best_score_val = scan(pooled, "observed", "expected", FEATURES, n_restarts=20, seed=42)

    sub_df = cells_in_subset(pooled, FEATURES, best_subset)
    C, B = sub_df["observed"].sum(), sub_df["expected"].sum()
    q_hat = C / B if B > 0 else float("nan")

    print(f"\nTop subset ({len(best_subset)}/{len(FEATURES)} features restricted):")
    for f in FEATURES:
        if f in best_subset:
            print(f"  {f}: {best_subset[f]}")
        else:
            print(f"  {f}: (unrestricted)")
    print(f"\nScore F = {best_score_val:.2f}")
    print(f"C (observed) = {C:.0f}, B (expected) = {B:.2f}, q_hat = C/B = {q_hat:.3f}")
    print(f"Cells covered: {len(sub_df)}/{len(pooled)}; claims covered: {C:.0f}/{pooled['observed'].sum()}")

    # Which windows actually contributed the excess for this subset's cells?
    detail = df.merge(sub_df[FEATURES], on=FEATURES, how="inner")
    per_window = detail.groupby("window_label", observed=True).agg(observed=("observed", "sum"), expected=("expected", "sum"))
    per_window["excess"] = per_window["observed"] - per_window["expected"]
    contributing_windows = set(per_window[per_window["excess"] > 0].index)
    falls_entirely_in_reduced_confidence = contributing_windows.issubset(REDUCED_CONFIDENCE_WINDOWS) and len(contributing_windows) > 0
    print(f"\nPer-window breakdown of this subset's excess (observed - expected):")
    print(per_window.to_string())
    print(
        f"\nFlag: top subset's excess falls entirely within reduced-confidence 2022 windows "
        f"({', '.join(sorted(REDUCED_CONFIDENCE_WINDOWS))})? "
        f"{'YES — treat this finding with caution' if falls_entirely_in_reduced_confidence else 'NO — excess is spread across other windows too'}"
    )

    return {
        "subset": best_subset, "score": best_score_val, "C": C, "B": B, "q_hat": q_hat,
        "n_cells": len(sub_df), "n_pooled_cells": len(pooled),
        "falls_entirely_in_reduced_confidence": falls_entirely_in_reduced_confidence,
    }


# =============================================================================
# PART D — summary
# =============================================================================

def print_summary(all_pass, results, restart_fraction, real_scan_result):
    print("\n" + "=" * 78)
    print("PART D — SUMMARY")
    print("=" * 78)

    print("\nPart B validation:")
    for name, ok in results.items():
        print(f"  {name:20s} {'PASS' if ok else 'FAIL'}")
    print(f"  Restart recovery fraction (informational, toy decoy problem): {restart_fraction:.0%}")

    if not all_pass:
        print("\nPart B did NOT all-pass — Part C was skipped. Fix the implementation before proceeding.")
        return

    r = real_scan_result
    print(f"\nPart C real-scan smoke test:")
    print(f"  Top subset restricts {len(r['subset'])}/{len(FEATURES)} features")
    print(f"  Score F = {r['score']:.2f}, C={r['C']:.0f}, B={r['B']:.2f}, q_hat={r['q_hat']:.3f}")
    print(f"  Covers {r['n_cells']}/{r['n_pooled_cells']} cells, {r['C']:.0f} claims")
    print(f"  Falls entirely in reduced-confidence 2022 windows: {r['falls_entirely_in_reduced_confidence']}")

    print(
        "\nThis script produces only RANK-1 subsets and single scores. Top-k iterative "
        "rescanning (finding the 2nd, 3rd, ... best non-overlapping subsets) and "
        "p-value calibration (via the negative-binomial null, randomization/parametric "
        "bootstrap over many null replicas) are deliberately deferred to later scripts "
        "(E1 calibration, E2 injection, E3 discovery). The score reported here is NOT "
        "a p-value and should not be interpreted as statistical significance yet."
    )


def main():
    processed_md5_before = mod04.file_md5(config.PROCESSED_DATA_PATH)
    active_md5_before = mod04.file_md5(config.ACTIVE_DATA_PATH)
    anchored_md5_before = mod04.file_md5(config.ANCHORED_EXPECTED_COUNTS_PATH)

    all_pass, results, restart_fraction = run_validation()

    real_scan_result = None
    if all_pass:
        real_scan_result = run_real_scan()

    print_summary(all_pass, results, restart_fraction, real_scan_result)

    processed_md5_after = mod04.file_md5(config.PROCESSED_DATA_PATH)
    active_md5_after = mod04.file_md5(config.ACTIVE_DATA_PATH)
    anchored_md5_after = mod04.file_md5(config.ANCHORED_EXPECTED_COUNTS_PATH)
    print("\n" + "=" * 78)
    print(f"Source unchanged — processed_data.csv: {processed_md5_before == processed_md5_after} (MD5 {processed_md5_after})")
    print(f"Source unchanged — {config.ACTIVE_DATA_PATH.name}: {active_md5_before == active_md5_after} (MD5 {active_md5_after})")
    print(f"Source unchanged — {config.ANCHORED_EXPECTED_COUNTS_PATH.name}: {anchored_md5_before == anchored_md5_after} (MD5 {anchored_md5_after})")


if __name__ == "__main__":
    main()
