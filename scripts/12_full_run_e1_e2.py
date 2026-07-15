"""Driver / orchestrator for the DEFINITIVE, internally-consistent M=999 pass of
E1 + the full four-method E2 comparison. One entry point; runs the dependency
chain in order and performs every integrity/fairness check, so the final numbers
are reproducible from here.

THIS SCRIPT ONLY ORCHESTRATES. It imports and CALLS the existing scripts'
functions — it never re-implements or forks their statistical logic. The harness
core (08) and every method script (07/09/10/11) are unmodified (06 was already
minimally parameterized for FGSS in a prior task; nothing here touches it). MD5s
of all source artifacts are checked before/after.

DEPENDENCY ORDER
  A. E1 null at M=999 (reuse mod07.run_window_nulls / run_calibration_check) ->
     rewrite reports/e1_null_maxscores.csv; adjudicate 2024-Q1.
  B. Proposed Poisson-MDSS E2, scored against the NEW M=999 E1 null
     (reuse mod08.run_step_experiment / run_ramp_experiment). Precision is DERIVED
     from the stored Jaccard+recall (exact algebra; see derive_precision) because
     08 predates the precision metric and must not be modified.
  C. Moving-average MDSS at M=999, K FIXED = 6 (no K re-selection) — its OWN null
     (reuse mod09.calibrate_ma(6,...) / run_ma_step_experiment / run_ma_ramp_experiment).
  D. FP-growth EPM — already M=999 (keep its step rows); re-run only its RAMP at the
     bumped trial count (reuse mod10 functions, its own null from disk).
  E. FGSS — already M=999 (keep its step rows); re-run only its RAMP at the bumped
     count (reuse mod11 + mod08.run_ramp_experiment, its own null from disk).

Each method is scored ONLY against its OWN M=999 null: proposed = E1's null;
MA/EPM/FGSS = their own (e2_*_null_*.csv). No baseline reads the proposed null.

RAMP TRIAL COUNT: bumped to RAMP_TRIALS (Option 1 — adequately-powered
time-to-detect for the FGSS-crossover phase-of-emergence finding). ALL FOUR ramps
are re-run at the same RAMP_TRIALS so time-to-detect is uniform. The step grid is
unchanged.

SUBGROUP CAVEAT (documented, honest): each method script draws its OWN injection
subgroups (its run_*_experiment seeds the subgroup RNG with its own method name,
by design — not modified here). So trials are matched by DESIGN CELL
(experiment, j, r, window, trial-index, arm), not by identical subgroup content.
The common-trial headline table therefore controls the window/difficulty MIX and
gives each method the same design, but the exact injected subgroup at a slot
differs across methods (a per-method randomized comparison over one common
design). Differences in each method's trial set vs the intersection are due to
per-method MIN_BASELINE_EXPECTED validity skips (different subgroups skip
different windows). Read-only on data/.
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

import config


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_scripts_dir = Path(__file__).resolve().parent
mod04 = _load_module("mod04", _scripts_dir / "04_expected_counts_fit.py")
mod07 = _load_module("e1_07", _scripts_dir / "07_e1_calibration.py")
mod08 = _load_module("mod08", _scripts_dir / "08_e2_injection.py")
mod09 = _load_module("mod09", _scripts_dir / "09_moving_average_mdss.py")
mod10 = _load_module("mod10", _scripts_dir / "10_fpgrowth_epm.py")
mod11 = _load_module("mod11", _scripts_dir / "11_fgss.py")

FEATURES = mod08.FEATURES
J_GRID = mod08.J_GRID
R_GRID = mod08.R_GRID
N_TRIALS_STEP = mod08.N_TRIALS_FULL
N_RESTARTS_GRID = mod08.N_RESTARTS_GRID
N_RESTARTS_NULL = mod07.N_RESTARTS_NULL
RAMP_LENGTH = mod08.RAMP_LENGTH
MIN_BASELINE_EXPECTED = mod08.MIN_BASELINE_EXPECTED
BASE_SEED = mod08.BASE_SEED
FLAG_WINDOWS = mod08.FLAG_WINDOWS

# ---------------------------------------------------------------------------
# RUN CONFIG
# ---------------------------------------------------------------------------
DRY_RUN = False  # True: tiny params, writes ONLY to scratch; validates wiring.
M_FULL = 999
RAMP_TRIALS = 50           # Option 1: bumped from 20 for adequately-powered time-to-detect.
K_MA_MONTHS = 6            # FIXED (already selected in 09); no re-selection here.
MISCAL_HIGH = mod07.MISCALIBRATION_HIGH  # 0.10

# Seeds — reuse each script's own seed_base so results reproduce its canonical run.
SEED_POIS_STEP = BASE_SEED + 1
SEED_POIS_RAMP = BASE_SEED + 2
SEED_MA_CAL = BASE_SEED + 700
SEED_MA_STEP = BASE_SEED + 800
SEED_MA_RAMP = BASE_SEED + 900
SEED_EPM_RAMP = BASE_SEED + 1700
SEED_FGSS_RAMP = BASE_SEED + 2100

STD_COLS = ["method", "j", "r", "window", "trial", "arm", "S", "detected", "score",
            "pvalue", "jaccard", "recall", "precision"]

if DRY_RUN:
    M = 15
    RAMP_TRIALS = 2
    N_TRIALS_STEP = 2
    N_RESTARTS_GRID = 2
    N_RESTARTS_NULL = 2
    OUT_DIR = Path(config.REPORTS_DIR).parent / "_scratch_driver_dryrun"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
else:
    M = M_FULL
    OUT_DIR = config.REPORTS_DIR

STEP_CSV = OUT_DIR / "e2_step_results.csv"
RAMP_CSV = OUT_DIR / "e2_ramp_results.csv"
E1_CSV = OUT_DIR / "e1_null_maxscores.csv"


# ===========================================================================
# derived precision (exact algebra from stored Jaccard + recall)
# ===========================================================================

def derive_precision(J, R, detected):
    """precision = |A∩B|/|A|. From J=|A∩B|/|A∪B| and R=|A∩B|/|B|:
        precision = R*J / (R - J + R*J).
    Exact whenever the recovered set A is nonempty. At J=R=0 the recovered set is
    either disjoint-nonempty (precision 0) or empty (precision NaN); a DETECTED
    trial always has a positive-scoring, nonempty A, so J=R=0 => disjoint => 0.
    For a non-detected trial we cannot tell empty from disjoint, so return NaN
    (affects only mean_precision_all, never the detected-trial headline)."""
    if pd.isna(J) or pd.isna(R):
        return np.nan
    denom = R - J + R * J
    if denom > 1e-12:
        return R * J / denom
    return 0.0 if detected else np.nan


def add_derived_precision(step_df):
    df = step_df.copy()
    prec = []
    for _, row in df.iterrows():
        if row["arm"] != "injected":
            prec.append(np.nan)
        else:
            prec.append(derive_precision(row["jaccard"], row["recall"], bool(row["detected"])))
    df["precision"] = prec
    return df


def load_method_null(csv_path, score_col, window_col="window_label"):
    """Reconstruct {window -> np.array of null max-statistics} from a saved
    per-method null CSV (used to re-run a method's ramp without recalibrating)."""
    df = pd.read_csv(csv_path)
    return {w: g[score_col].to_numpy() for w, g in df.groupby(window_col)}


# ===========================================================================
# STEP A — E1 at M=999
# ===========================================================================

def run_e1_full(alpha):
    print("\n" + "=" * 78)
    print(f"STEP A — E1 NULL CALIBRATION at M={M} (reusing mod07 functions)")
    print("=" * 78)
    df_anchored, windows = mod07.load_anchored_counts()
    null_by_window, check_fractions, long_rows = {}, {}, []
    for w_idx, w in enumerate(windows):
        window_df = df_anchored[df_anchored["window_label"] == w][FEATURES + ["observed", "expected"]].reset_index(drop=True)
        seed_main = mod07.BASE_SEED + 1000 * w_idx
        seed_check = mod07.BASE_SEED + 1000 * w_idx + 500
        max_scores, _bad, wt = mod07.run_window_nulls(window_df, alpha, M, N_RESTARTS_NULL, seed_main)
        p95 = np.percentile(max_scores, 95)
        null_by_window[w] = max_scores
        for i, s in enumerate(max_scores):
            long_rows.append({"window_label": w, "replica_idx": i, "max_score": s})
        exceed, _cs, _cb = mod07.run_calibration_check(window_df, alpha, p95, M, N_RESTARTS_NULL, seed_check)
        check_fractions[w] = exceed
        print(f"  [{w}] total={wt} p95={p95:.2f} calibration-check FPR={exceed:.1%}")

    pd.DataFrame(long_rows).to_csv(E1_CSV, index=False)
    pooled = float(np.mean(list(check_fractions.values())))
    print(f"\nWrote {E1_CSV} ({len(long_rows)} rows). Pooled M={M} calibration FPR = {pooled:.1%} (target 5%).")

    # --- adjudicate 2024-Q1 ---
    q1 = check_fractions.get("2024-Q1", np.nan)
    verdict_high = q1 > MISCAL_HIGH
    print(f"\n2024-Q1 ADJUDICATION (M=199 had flagged it at 11.1%): M={M} calibration-check FPR = {q1:.1%}.")
    if verdict_high:
        print(f"  VERDICT: still > {MISCAL_HIGH:.0%} -> genuinely reduced-confidence; carry a flag for 2024-Q1 into E2/E3.")
    else:
        print(f"  VERDICT: settles at {q1:.1%} (<= {MISCAL_HIGH:.0%}) -> the M=199 flag was small-sample noise; 2024-Q1 is adequately calibrated.")
    return null_by_window, check_fractions, windows, pooled, {"q1_fpr": q1, "q1_reduced_confidence": bool(verdict_high)}


# ===========================================================================
# STEP B — proposed Poisson-MDSS against the M=999 E1 null
# ===========================================================================

def run_poisson(base_data, e1_null, windows, alpha):
    print("\n" + "=" * 78)
    print("STEP B — PROPOSED Poisson-MDSS E2 (scored vs M=999 E1 null)")
    print("=" * 78)
    step = mod08.run_step_experiment(
        [mod08.POISSON_MDSS], base_data, e1_null, windows, J_GRID, R_GRID,
        N_TRIALS_STEP, N_RESTARTS_GRID, SEED_POIS_STEP, MIN_BASELINE_EXPECTED, alpha,
    )
    step = add_derived_precision(step)  # 08 stores Jaccard+recall only; precision derived exactly for detected
    ramp = mod08.run_ramp_experiment(
        [mod08.POISSON_MDSS], base_data, e1_null, windows, J_GRID, R_GRID,
        RAMP_TRIALS, N_RESTARTS_GRID, RAMP_LENGTH, SEED_POIS_RAMP, MIN_BASELINE_EXPECTED, alpha,
    )
    print(f"Poisson: {len(step)} step rows, {len(ramp)} ramp rows.")
    return step, ramp


# ===========================================================================
# STEP C — moving-average MDSS, M=999, K fixed = 6
# ===========================================================================

def run_ma(infra):
    print("\n" + "=" * 78)
    print(f"STEP C — MOVING-AVERAGE MDSS at M={M}, K FIXED = {K_MA_MONTHS} (no re-selection)")
    print("=" * 78)
    k_windows = K_MA_MONTHS // 3
    est_scans = len(infra["windows"]) * M * 2 * N_RESTARTS_NULL
    print(f"MA calibration (own null, K={K_MA_MONTHS}): ~{est_scans:,} scan calls. This is the compute long pole.")
    t0 = time.time()
    null_results, check_fractions, long_df = mod09.calibrate_ma(K_MA_MONTHS, infra, M, N_RESTARTS_NULL, SEED_MA_CAL, verbose=True)
    ma_null = {w: null_results[w]["scores"] for w in infra["windows"]}
    print(f"MA calibration done in {(time.time()-t0)/60:.1f} min. Pooled FPR = {np.mean(list(check_fractions.values())):.1%}")

    # persist the K=6 M=999 null (preserve any prior K=3 selection rows for provenance)
    ma_null_path = config.REPORTS_DIR / "e2_ma_null_maxscores.csv"
    long_df = long_df.copy()
    if not DRY_RUN and ma_null_path.exists():
        prior = pd.read_csv(ma_null_path)
        prior = prior[prior["K"] != K_MA_MONTHS]
        pd.concat([prior, long_df], ignore_index=True).to_csv(ma_null_path, index=False)
    step = mod09.run_ma_step_experiment(infra, ma_null, k_windows, J_GRID, R_GRID, N_TRIALS_STEP, N_RESTARTS_GRID, SEED_MA_STEP, MIN_BASELINE_EXPECTED)
    ramp = mod09.run_ma_ramp_experiment(infra, ma_null, k_windows, J_GRID, R_GRID, RAMP_TRIALS, N_RESTARTS_GRID, SEED_MA_RAMP, MIN_BASELINE_EXPECTED)
    print(f"MA: {len(step)} step rows, {len(ramp)} ramp rows.")
    return step, ramp, check_fractions, ma_null


# ===========================================================================
# STEP D/E — EPM & FGSS ramps re-run at the bumped count (steps kept from disk)
# ===========================================================================

def run_epm_ramp(infra):
    print("\n" + "=" * 78)
    print(f"STEP D — EPM RAMP re-run at RAMP_TRIALS={RAMP_TRIALS} (step rows kept; own M=999 null from disk)")
    print("=" * 78)
    item_cols, col_of, onehot = mod10.build_epm_index(infra["cells_613"])
    hp = mod10.fix_epm_hyperparams(infra)
    epm = {"item_cols": item_cols, "col_of": col_of, "onehot": onehot,
           "minsup": hp["minsup"], "smooth_floor": hp["smooth_floor"], "mu_anchored": mod10.precompute_mu_anchored(infra)}
    epm_null = load_method_null(config.REPORTS_DIR / "e2_epm_null_maxratios.csv", "max_growth_ratio")
    ramp = mod10.run_epm_ramp_experiment(infra, epm, epm_null, J_GRID, R_GRID, RAMP_TRIALS, SEED_EPM_RAMP, MIN_BASELINE_EXPECTED)
    print(f"EPM ramp: {len(ramp)} rows.")
    return ramp


def run_fgss_ramp(base_data, windows, alpha):
    print("\n" + "=" * 78)
    print(f"STEP E — FGSS RAMP re-run at RAMP_TRIALS={RAMP_TRIALS} (step rows kept; own M=999 null from disk)")
    print("=" * 78)
    method = mod11.make_fgss_method(alpha)
    fgss_null = load_method_null(config.REPORTS_DIR / "e2_fgss_null_maxscores.csv", "max_bj_score")
    ramp = mod08.run_ramp_experiment(
        [method], base_data, fgss_null, windows, J_GRID, R_GRID,
        RAMP_TRIALS, N_RESTARTS_GRID, RAMP_LENGTH, SEED_FGSS_RAMP, MIN_BASELINE_EXPECTED, alpha,
    )
    print(f"FGSS ramp: {len(ramp)} rows.")
    return ramp


# ===========================================================================
# INTEGRITY + FAIRNESS
# ===========================================================================

def step_trial_key(df):
    return set(zip(df["j"], df["r"], df["window"], df["trial"], df["arm"]))


def integrity_and_headline(step_all, ramp_all, e1_null, ma_null_windows):
    print("\n" + "=" * 78)
    print("INTEGRITY + FAIRNESS CHECKS")
    print("=" * 78)
    methods = sorted(step_all["method"].unique())

    # 1. row counts + duplicate trial keys
    print("\n[1] Row counts + duplicate-key check:")
    for m in methods:
        s = step_all[step_all["method"] == m]
        r = ramp_all[ramp_all["method"] == m]
        dup = int(s.duplicated(["j", "r", "window", "trial", "arm"]).sum())
        dup_r = int(r.duplicated(["j", "r", "trial"]).sum())
        print(f"  {m:22s} step={len(s):5d} (dupes {dup})  ramp={len(r):4d} (dupes {dup_r})")
        assert dup == 0 and dup_r == 0, f"duplicate trial keys for {m}"

    # 2. trial-key alignment (fairness)
    print("\n[2] Trial-key alignment (key = j,r,window,trial,arm; subgroups differ per method — see header):")
    keysets = {m: step_trial_key(step_all[step_all["method"] == m]) for m in methods}
    common = set.intersection(*keysets.values())
    print(f"  Common trial slots across all {len(methods)} methods: {len(common)}")
    for m in methods:
        print(f"    {m:22s} has {len(keysets[m]):5d} slots ({len(keysets[m]) - len(common):+d} beyond intersection — MIN_BASELINE skips)")

    # 3. null independence
    print("\n[3] Null independence — each method scored vs its OWN M=999 null:")
    print(f"    proposed Poisson-MDSS -> E1 null (n_replicas/window ~ {len(next(iter(e1_null.values())))})")
    print(f"    moving_average_mdss   -> e2_ma_null_maxscores.csv (K={K_MA_MONTHS}, ~{len(next(iter(ma_null_windows.values())))})")
    print(f"    fpgrowth_epm          -> e2_epm_null_maxratios.csv (own)")
    print(f"    fgss                  -> e2_fgss_null_maxscores.csv (own)")
    assert not np.array_equal(np.sort(next(iter(e1_null.values()))),
                              np.sort(next(iter(ma_null_windows.values())))), "proposed and MA nulls must differ"
    print("    OK: proposed null != MA null (distinct distributions); no baseline reads the proposed null.")

    # headline on the COMMON slot set
    common_df = step_all[step_all.apply(lambda r: (r["j"], r["r"], r["window"], r["trial"], r["arm"]) in common, axis=1)]
    rows = []
    for (m, j, r), g in common_df.groupby(["method", "j", "r"]):
        clean = g[g["arm"] == "clean"]
        inj = g[g["arm"] == "injected"]
        det = inj[inj["detected"]]
        rows.append({
            "method": m, "j": j, "r": r,
            "fpr": clean["detected"].mean(), "power": inj["detected"].mean(),
            "precision_detected": det["precision"].mean(), "recall_detected": det["recall"].mean(),
            "jaccard_detected": det["jaccard"].mean(), "n_common_injected": len(inj),
        })
    headline = pd.DataFrame(rows)

    # ramp median time-to-detect on common (j,r,trial) slots
    ramp_methods = sorted(ramp_all["method"].unique())
    rkeys = {m: set(zip(ramp_all[ramp_all.method == m]["j"], ramp_all[ramp_all.method == m]["r"], ramp_all[ramp_all.method == m]["trial"])) for m in ramp_methods}
    rcommon = set.intersection(*rkeys.values()) if rkeys else set()
    rc = ramp_all[ramp_all.apply(lambda x: (x["j"], x["r"], x["trial"]) in rcommon, axis=1)]
    ramp_headline = []
    for (m, j, r), g in rc.groupby(["method", "j", "r"]):
        det = g[~g["censored"]]
        ramp_headline.append({"method": m, "j": j, "r": r, "n_ramp_common": len(g),
                              "n_detected": len(det), "median_ttd": det["time_to_detect"].median() if len(det) else np.nan})
    ramp_headline = pd.DataFrame(ramp_headline)
    print(f"\n  Ramp common (j,r,trial) slots across methods: {len(rcommon)}; per-cell detected counts in headline below.")
    return headline, ramp_headline, len(common)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    t_start = time.time()
    md5_paths = {
        "processed_data.csv": config.PROCESSED_DATA_PATH,
        config.ACTIVE_DATA_PATH.name: config.ACTIVE_DATA_PATH,
        config.ANCHORED_EXPECTED_COUNTS_PATH.name: config.ANCHORED_EXPECTED_COUNTS_PATH,
    }
    md5_before = {n: mod04.file_md5(p) for n, p in md5_paths.items()}
    mod08_md5_before = mod04.file_md5(_scripts_dir / "08_e2_injection.py")

    print(f"{'DRY RUN (scratch outputs, tiny params)' if DRY_RUN else 'FULL RUN (M=999)'}: "
          f"M={M}, RAMP_TRIALS={RAMP_TRIALS}, N_TRIALS_STEP={N_TRIALS_STEP}, out={OUT_DIR}")

    # keep EPM + FGSS M=999 STEP rows from the current results file (read BEFORE overwrite)
    prior_step = pd.read_csv(config.REPORTS_DIR / "e2_step_results.csv")
    kept_step = prior_step[prior_step["method"].isin(["fpgrowth_epm", "fgss"])].copy()
    for c in STD_COLS:
        if c not in kept_step.columns:
            kept_step[c] = np.nan
    print(f"Kept from disk (already M=999): "
          f"{ {m: int((kept_step['method']==m).sum()) for m in ['fpgrowth_epm','fgss']} } step rows.")

    alpha = mod07.recover_alpha()

    # A
    e1_null, e1_checks, windows, e1_pooled, q1 = run_e1_full(alpha)

    # base_data for the proposed method; override its null with the fresh in-memory M=999 E1 null
    base_data, _old_null, _w = mod08.load_base_data()

    # B
    pois_step, pois_ramp = run_poisson(base_data, e1_null, windows, alpha)

    # C
    infra = mod09.build_infrastructure()
    ma_step, ma_ramp, ma_checks, ma_null_windows = run_ma(infra)

    # D, E — ramps only
    epm_ramp = run_epm_ramp(infra)
    fgss_ramp = run_fgss_ramp(base_data, windows, alpha)

    # assemble
    step_all = pd.concat([pois_step[STD_COLS], ma_step[STD_COLS], kept_step[STD_COLS]], ignore_index=True)
    ramp_cols = ["method", "j", "r", "trial", "onset_window", "S", "time_to_detect", "censored"]
    ramp_all = pd.concat([pois_ramp[ramp_cols], ma_ramp[ramp_cols], epm_ramp[ramp_cols], fgss_ramp[ramp_cols]], ignore_index=True)
    step_all.to_csv(STEP_CSV, index=False)
    ramp_all.to_csv(RAMP_CSV, index=False)
    print(f"\nWrote {STEP_CSV} ({len(step_all)}) and {RAMP_CSV} ({len(ramp_all)}).")

    headline, ramp_headline, n_common = integrity_and_headline(step_all, ramp_all, e1_null, ma_null_windows)

    # per-method full summaries (not intersection) for reference + plots
    step_summary = mod11.summarize_fgss_step(step_all)  # generic: groups by method,j,r with precision
    ramp_summary = mod08.summarize_ramp(ramp_all)
    step_summary.to_csv(OUT_DIR / "e2_step_summary.csv", index=False)
    ramp_summary.to_csv(OUT_DIR / "e2_ramp_summary.csv", index=False)
    headline.to_csv(OUT_DIR / "e2_headline_common_trials.csv", index=False)
    ramp_headline.to_csv(OUT_DIR / "e2_headline_ramp_common.csv", index=False)

    if not DRY_RUN:
        # regenerate 4-method plots at M=999 (reuse mod09's by-method plotters)
        mod09.plot_power_curves_by_method(step_summary, "e2_power_curves_by_method.png")
        mod09.plot_precision_recall_jaccard(step_summary, "e2_precision_recall_jaccard_by_method.png")
        mod09.plot_time_to_detect_by_method(ramp_summary, "e2_time_to_detect_by_method.png")

    # ---- summary ----
    print("\n" + "=" * 78)
    print("FINAL SUMMARY — E1 + E2 DEFINITIVE M=999")
    print("=" * 78)
    print(f"\n2024-Q1 verdict: calibration-check FPR = {q1['q1_fpr']:.1%} -> "
          f"{'REDUCED-CONFIDENCE (flag carried to E2/E3)' if q1['q1_reduced_confidence'] else 'adequately calibrated (M=199 flag was noise)'}.")
    print(f"E1 pooled M={M} calibration FPR = {e1_pooled:.1%}. Per-window:")
    for w in windows:
        print(f"    {w:9s} {e1_checks[w]:.1%}" + ("  <FLAG" if w in FLAG_WINDOWS else ""))
    print(f"\nRamp decision: Option 1 (bumped RAMP_TRIALS={RAMP_TRIALS}); all four methods re-run uniformly. "
          "Per-(j,r) detected counts in the ramp headline:")
    print(ramp_headline.to_string(index=False))
    print(f"\nTrial-key alignment: {n_common} common step slots (paired by design cell; subgroups differ per method).")
    print("\nHEADLINE 4-way comparison on the COMMON trial set (power/FPR/precision/recall/Jaccard):")
    print(headline.to_string(index=False))
    print("\n4-way power pivot (common trials):")
    print(headline.pivot_table(index=["j", "r"], columns="method", values="power").to_string())

    mod08_md5_after = mod04.file_md5(_scripts_dir / "08_e2_injection.py")
    print(f"\nHarness core 08 unchanged: {mod08_md5_before == mod08_md5_after}")
    print("Source artifact integrity:")
    for n, before in md5_before.items():
        print(f"  {n}: unchanged={before == mod04.file_md5(md5_paths[n])}")
    print(f"\nEvery method scored vs its OWN M={M} null. E1 and E2 are now final and consistent. "
          "E3 (discovery) is the remaining experiment — NOT run here.")
    print(f"\nTotal driver time: {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
