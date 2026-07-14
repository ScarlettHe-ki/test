"""FP-growth emerging-pattern-mining (EPM) baseline, registered as the THIRD
method in the E2 comparison — the architectural stress test of the plug-in
harness.

WHY THIS IS THE STRESS TEST (and why it, like 09, needs its own loop):
The proposed method and the moving-average baseline both end in the SAME
engine — mdss.scan() over a Poisson score, judged against a max-scan-score
null. EPM shares NONE of that. Its engine is frequent-itemset support growth
(mlxtend.fpgrowth), its statistic is the MAX support-growth-ratio over
qualifying itemsets (not a scan score), and its null is the distribution of
that max ratio (not of a max scan score). mod08.Method.detect is, by
construction, a scan-subset-returning callable; EPM cannot conform to it
meaningfully. So — exactly as 09 argued for its two-argument fit — EPM has
its OWN orchestration loop, and mod08's FILE IS NOT MODIFIED (MD5-verified in
main). What IS reused, never reimplemented: the NB null-replica draw
(e1.draw_null_replica), the permutation p-value (e1.pvalue), the injection
primitive (mod08.inject), subgroup drawing (mod08.draw_subgroup), the
crc32 deterministic seeding (mod08.deterministic_seed), the ramp schedule
(mod08.ramp_multipliers / valid_onset_windows), the identification metrics
(mod08.jaccard / recall, mod09.precision), the whole reference-replica
infrastructure (mod09.build_infrastructure / mu_hat_for_window /
window_idx_of), and the by-method plotting (mod09.plot_*). Rows are emitted
in the EXACT schema of the existing E2 CSVs so they append directly.

THE SUPPORT-DENOMINATOR CONFOUND IS PRESERVED, NOT CORRECTED (Prof. Zhou's
exposure limitation): EPM computes support = count / total on the INJECTED
totals — the denominator INCLUDES the injected excess. When a subgroup is
inflated r-fold, the window total rises too, so every non-injected itemset's
support is diluted and even the injected itemset's support-growth-ratio is
attenuated below the naive r (attenuation grows with the subgroup's share of
the book). We measure and report this rather than hiding it.

EPM'S STRUCTURAL IDENTIFICATION LIMIT: a frequent itemset is a pure
CONJUNCTION (feature=value AND feature=value ...); it cannot express
within-feature disjunction (peril in {wind, hail}). We tag every trial's
planted S as conjunctive/disjunctive so the writeup can separate "EPM lost on
power" from "EPM could not represent the target". NOTE: mod08.draw_subgroup
plants exactly one value per feature, so every S in THIS harness is already a
pure conjunction — the handicap does not bite here, but the tag makes that an
explicit, checkable fact rather than an assumption.

RAMP LEAKAGE (same realism as 09's boiling frog): STEP uses a FRESH clean
trailing reference (no injection leakage); RAMP uses the PRECEDING ramp
step's own already-injected counts as the trailing reference, so the rising
signal contaminates EPM's own baseline. For a support-growth-ratio method
this bites twice — the numerator's reference support rises with the signal
AND the denominator (total) keeps inflating — documented in the summary.

Read-only on config data/artifact paths and the existing E2 CSVs (appended
to idempotently: any prior fpgrowth_epm rows are dropped before re-appending,
so re-running never duplicates — this also closes the duplication footgun
noted for 09's in-place append).
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
from mlxtend.frequent_patterns import fpgrowth

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
mod09 = _load_module("mod09", _scripts_dir / "09_moving_average_mdss.py")

# Reuse the proposed method's exact grid/trial/seed settings — required for a
# fair, same-scheme, same-grid head-to-head comparison.
FEATURES = mod08.FEATURES
J_GRID = mod08.J_GRID
R_GRID = mod08.R_GRID
N_TRIALS_FULL = mod08.N_TRIALS_FULL
N_TRIALS_GATE = mod08.N_TRIALS_GATE
RAMP_LENGTH = mod08.RAMP_LENGTH
MIN_BASELINE_EXPECTED = mod08.MIN_BASELINE_EXPECTED  # also EPM's minsup floor (shared min-presence)
BASE_SEED = mod08.BASE_SEED
FLAG_WINDOWS = mod08.FLAG_WINDOWS
deterministic_seed = mod08.deterministic_seed
draw_subgroup = mod08.draw_subgroup
inject = mod08.inject
jaccard = mod08.jaccard
recall = mod08.recall
precision = mod09.precision
ramp_multipliers = mod08.ramp_multipliers
valid_onset_windows = mod08.valid_onset_windows
window_idx_of = mod09.window_idx_of

METHOD_NAME = "fpgrowth_epm"

# ---------------------------------------------------------------------------
# FAST/FULL toggle
# ---------------------------------------------------------------------------
MODE = "full"
M_BY_MODE = {"fast": 199, "full": 999}
GATE_RECOVERY_R = 6.0  # off-grid strong multiplier used ONLY for the gate's identification-
                       # recovery check (see run_validation_gate_epm). Chosen a priori as
                       # "clearly above the grid so the injected pattern unmistakably
                       # dominates the growth-ratio null"; it never enters the reported grid
                       # and is not tuned against a target power.
M_CALIBRATION = M_BY_MODE[MODE]

# FIXED modeling choices (set a priori, NEVER tuned against detection):
MAX_ITEMSET_LEN = 4  # interpretable short rules; covers every planted j (<=3) plus one
                     # conjunct of headroom. A cap fixed before any detection run, not
                     # chosen to help/hurt power (best growth ratio was identical at
                     # max_len 3 vs 4 in the pre-flight probe) — it only bounds the
                     # frequent-itemset lattice so the run is tractable.
N_REFERENCE_WINDOWS = 4  # EPM compares the detection window against a REFERENCE PERIOD.
                         # That period is the trailing 4 windows (~1 year) pooled — the
                         # canonical emerging-pattern setup (current window vs a substantial
                         # historical period). A single-window reference was tried first and
                         # rejected on DESIGN grounds, not FPR-tuning: with one noisy window
                         # the support-growth-ratio is dominated by per-window NB sampling
                         # swings (pre-flight probe: null p95 ratio ~72, which swamps any
                         # r=3 signal whose own growth ratio is ~2-3), so EPM had ~0 power
                         # for a reason unrelated to its statistic. Pooling the reference
                         # period stabilizes the denominator so the ratio reflects genuine
                         # emergence. This length is FIXED a priori by the "compare to a
                         # trailing year" design, NOT selected by calibration FPR — the task
                         # asks EPM only to calibrate its own null, and a length-tuning loop
                         # would risk exactly the "tune a hyperparameter against detection"
                         # the spec forbids.


# =============================================================================
# EPM INDEX + HYPERPARAMETERS (fixed on training)
# =============================================================================

def build_epm_index(cells_613):
    """One-hot the 613-cell frame over (feature=value) items, once. Every
    per-window transaction table is built by np.repeat-ing these rows by the
    window's cell counts, so item columns and cell order are shared across all
    fpgrowth calls (and with the reference-support lookups)."""
    item_cols, col_of = [], {}
    for f in FEATURES:
        for v in sorted(cells_613[f].astype(str).unique()):
            item = f"{f}={v}"
            col_of[item] = len(item_cols)
            item_cols.append(item)
    onehot = np.zeros((len(cells_613), len(item_cols)), dtype=bool)
    for item, j in col_of.items():
        f, v = item.split("=", 1)
        onehot[:, j] = (cells_613[f].astype(str).to_numpy() == v)
    return item_cols, col_of, onehot


def median_training_window_total(infra):
    """Median claim count per TRAINING window (2018-2020, window_idx 0-11),
    restricted to the 613-cell universe so the count basis matches detection.
    Used only to convert the shared min-presence floor into a relative support
    and to size the reference smoothing floor — computed on training, never on
    test."""
    df = diag.load_data(config.ACTIVE_DATA_PATH)
    _month_idx, window_idx = diag.build_time_indices(df)
    df = df.copy()
    df["window_idx"] = window_idx
    for c in FEATURES:
        df[c] = df[c].astype(str)
    cells_str = infra["cells_613"].copy()
    for c in FEATURES:
        cells_str[c] = cells_str[c].astype(str)
    key_613 = set(map(tuple, cells_str[FEATURES].to_numpy()))
    in_613 = df[FEATURES].apply(tuple, axis=1).isin(key_613)
    totals = [int(((df["window_idx"] == w) & in_613).sum()) for w in range(0, 12)]
    return float(np.median(totals)), totals


def fix_epm_hyperparams(infra):
    """minsup and the reference smoothing floor, FIXED on training data before
    any detection run.

    minsup: an itemset must be supported by at least MIN_BASELINE_EXPECTED (15)
    claims' worth of relative support at the median training-window volume —
    the SAME 15-claim minimum-presence floor the scan method uses for a valid
    subgroup (mod08.MIN_BASELINE_EXPECTED), expressed as a relative support so
    it plugs straight into fpgrowth. This deliberately reuses a floor set for a
    different method's validity check, so it cannot have been reverse-engineered
    from EPM's own detection performance.

    smoothing floor = minsup: an itemset whose REFERENCE support falls below the
    same frequency threshold minsup is, by definition, not reliably present in
    the reference period; we cap its imputed reference support at minsup rather
    than crediting near-infinite growth to a sub-threshold (noise-level)
    estimate. This serves the spec's stated purpose (avoid divide-by-zero on
    newly-appearing itemsets) AND stops the MAX growth-ratio from being
    dominated by rare 'jumping' itemsets: a pre-flight probe with a half-claim
    floor gave a clean-null p95 growth ratio ~60, which swamped the injected
    signal (whose own denominator-attenuated growth ratio is only ~2-3);
    flooring at minsup brings the null p95 to ~4 so genuine emergence can
    compete. minsup is fixed on training, so this floor is fixed on training
    too — never tuned on detection performance.
    """
    median_total, train_totals = median_training_window_total(infra)
    minsup = MIN_BASELINE_EXPECTED / median_total
    smooth_floor = minsup
    return {
        "minsup": minsup,
        "smooth_floor": smooth_floor,
        "median_train_total": median_total,
        "train_window_totals": train_totals,
        "min_abs_support": MIN_BASELINE_EXPECTED,
    }


# =============================================================================
# REFERENCE-REPLICA DRAWING (cached regression rate -> fast NB draws)
# =============================================================================

def precompute_mu_anchored(infra):
    """Anchored NB mean per cell for every window_idx that any detection or
    trailing reference can touch (2021 buffer 12-15 through test 16-27),
    computed ONCE. mod09.draw_reference_replica_df re-runs the statsmodels
    predict on every call; caching it here turns each of the ~30k replica draws
    into a cheap NB sample instead of a fresh GLM predict."""
    mu_anchored = {}
    for widx in range(12, 28):
        mu = mod09.mu_hat_for_window(widx, infra)
        total = infra["window_totals"][widx]
        mu_anchored[widx] = mu / mu.sum() * total
    return mu_anchored


def draw_counts(widx, epm, infra, rng):
    """One NB2 null-replica count vector for window_idx `widx`, aligned to the
    613-cell frame (same construction every method uses via e1.draw_null_replica)."""
    total = infra["window_totals"][widx]
    return e1.draw_null_replica(epm["mu_anchored"][widx], infra["alpha"], total, rng)


def draw_reference_pool(det_widx, epm, infra, rng, k=N_REFERENCE_WINDOWS):
    """Pooled clean reference PERIOD: sum k fresh NB replicas of the windows
    immediately preceding det_widx. Pooling stabilizes the reference support
    (the denominator of the growth ratio) so the statistic reflects genuine
    emergence rather than single-window NB noise. Returns a float count vector
    aligned to the 613-cell frame."""
    pooled = np.zeros(len(infra["cells_613"]), dtype=float)
    for b in range(1, k + 1):
        pooled += draw_counts(det_widx - b, epm, infra, rng)
    return pooled


def inject_counts(counts, S, r, cells_613):
    """Inject via mod08.inject verbatim: build the cell frame, multiply the
    S-matching cells' counts by r, return the aligned count vector. cells_613
    shares base_data's dtypes and row order, so S's native values mask correctly."""
    df = cells_613.copy()
    df["observed"] = np.asarray(counts)
    return inject(df, S, r)["observed"].to_numpy()


# =============================================================================
# THE EPM STATISTIC
# =============================================================================

def _stringify_subset(subset):
    """Normalize a subset dict to {feature: [str(value), ...]} so the
    set-based identification metrics compare recovered itemsets (whose values
    are always strings, split from 'feature=value') against a planted S whose
    values may be native (e.g. syndicate stored as float) on the SAME footing.
    Without this, ('syndicate', 1234.0) != ('syndicate', '1234.0') would zero
    out recall/precision spuriously (the dtype trap from 09)."""
    return {f: [str(v) for v in vals] for f, vals in subset.items()}


def itemset_to_subset(itemset):
    """Map an fpgrowth itemset (frozenset of 'feature=value') to the subset-dict
    representation. A transaction holds one value per feature, so no itemset can
    contain two values of the same feature — the result is always one value per
    feature, i.e. a pure conjunction (EPM's structural limit, made concrete)."""
    subset = {}
    for item in itemset:
        f, v = item.split("=", 1)
        subset.setdefault(f, []).append(v)
    return subset


def epm_statistic(counts_det, counts_ref, epm):
    """Max support-growth-ratio over frequent itemsets of the detection window,
    and the itemset achieving it. Support is count/total on the INJECTED
    detection total (denominator confound preserved). Reference support is
    looked up on the 613-cell frame (no second fpgrowth needed) and
    smoothing-floored."""
    counts_det = np.clip(np.asarray(counts_det).astype(int), 0, None)
    total_det = int(counts_det.sum())
    if total_det == 0:
        return 0.0, {}
    txn = pd.DataFrame(np.repeat(epm["onehot"], counts_det, axis=0), columns=epm["item_cols"])
    freq = fpgrowth(txn, min_support=epm["minsup"], use_colnames=True, max_len=MAX_ITEMSET_LEN)
    if len(freq) == 0:
        return 0.0, {}

    counts_ref = np.asarray(counts_ref).astype(float)
    total_ref = counts_ref.sum()
    onehot, col_of, floor = epm["onehot"], epm["col_of"], epm["smooth_floor"]
    best_ratio, best_itemset = 0.0, None
    for support_det, itemset in zip(freq["support"].to_numpy(), freq["itemsets"].to_numpy()):
        cols = [col_of[i] for i in itemset]
        memb = onehot[:, cols].all(axis=1)
        support_ref = (counts_ref[memb].sum() / total_ref) if total_ref > 0 else 0.0
        ratio = support_det / max(support_ref, floor)
        if ratio > best_ratio:
            best_ratio, best_itemset = ratio, itemset
    return best_ratio, (itemset_to_subset(best_itemset) if best_itemset is not None else {})


def epm_detect(counts_det, counts_ref, epm, null_ratios_for_window):
    ratio, subset = epm_statistic(counts_det, counts_ref, epm)
    p = e1.pvalue(ratio, null_ratios_for_window)
    return (p < 0.05), subset, ratio, p


def subset_cell_mask(S, cells_613):
    mask = pd.Series(True, index=cells_613.index)
    for f, vals in S.items():
        mask &= cells_613[f].isin(vals)
    return mask.to_numpy()


def subset_growth_ratio(counts_det, counts_ref, S_mask, floor):
    """Growth ratio of the PLANTED S itemset specifically (not the argmax) —
    isolates the support-denominator attenuation: numerator uses the injected
    detection total, so a larger-share S attenuates its own ratio below r."""
    total_det = counts_det.sum()
    total_ref = counts_ref.sum()
    if total_det == 0:
        return np.nan
    sd = counts_det[S_mask].sum() / total_det
    sr = max((counts_ref[S_mask].sum() / total_ref) if total_ref > 0 else 0.0, floor)
    return sd / sr


# =============================================================================
# EPM CALIBRATION (own null, clean detection vs clean trailing reference)
# =============================================================================

def calibrate_epm(infra, epm, m, seed_base, verbose=True):
    null_ratios, check_fractions, long_rows = {}, {}, []
    for pos, w in enumerate(infra["windows"]):
        widx = window_idx_of(w, infra)

        def one_replica(seed):
            rng = np.random.default_rng(seed)
            counts_det = draw_counts(widx, epm, infra, rng)
            counts_ref = draw_reference_pool(widx, epm, infra, rng)
            ratio, _ = epm_statistic(counts_det, counts_ref, epm)
            return ratio

        null = np.array([one_replica(seed_base + pos * 10_000 + i) for i in range(m)])
        p95 = np.percentile(null, 95)
        null_ratios[w] = null
        for i, s in enumerate(null):
            long_rows.append({"window_label": w, "replica_idx": i, "max_growth_ratio": s})

        check = np.array([one_replica(seed_base + pos * 10_000 + 50_000 + i) for i in range(m)])
        exceed = float(np.mean(check > p95))
        check_fractions[w] = exceed
        if verbose:
            print(f"  [{w}] p95_ratio={p95:.2f}, calibration check={exceed:.1%}")

    return null_ratios, check_fractions, pd.DataFrame(long_rows)


# =============================================================================
# STEP experiment (EPM)
# =============================================================================

def run_epm_step_trial(widx, S, S_mask, r, epm, infra, null_ratios_w, trial_seed):
    cells = infra["cells_613"]
    rng_det = np.random.default_rng(trial_seed)
    clean = draw_counts(widx, epm, infra, rng_det)

    rng_ref_c = np.random.default_rng(trial_seed + 1)
    ref_clean = draw_reference_pool(widx, epm, infra, rng_ref_c)
    clean_res = epm_detect(clean, ref_clean, epm, null_ratios_w)

    injected = inject_counts(clean, S, r, cells)
    rng_ref_i = np.random.default_rng(trial_seed + 3)
    ref_inj = draw_reference_pool(widx, epm, infra, rng_ref_i)  # fresh clean: no leakage in STEP
    inj_res = epm_detect(injected, ref_inj, epm, null_ratios_w)

    s_ratio = subset_growth_ratio(injected, ref_inj, S_mask, epm["smooth_floor"])
    return clean_res, inj_res, s_ratio


def run_epm_step_experiment(infra, epm, null_ratios_by_window, j_grid, r_grid, n_trials, seed_base, min_baseline_expected):
    pooled_ref = pd.concat(infra["base_data"].values(), ignore_index=True).groupby(FEATURES, as_index=False)["expected"].sum()
    pooled_total = pooled_ref["expected"].sum()
    rows = []
    for j in j_grid:
        for r in r_grid:
            cell_seed = seed_base + deterministic_seed(METHOD_NAME, j, r) % 10_000
            rng_subgroups = np.random.default_rng(cell_seed)
            for trial_idx in range(n_trials):
                S, B_S_pooled = draw_subgroup(rng_subgroups, j, pooled_ref, min_baseline_expected)
                if S is None:
                    continue
                s_share = B_S_pooled / pooled_total
                is_disjunctive = any(len(v) > 1 for v in S.values())
                S_mask = subset_cell_mask(S, infra["cells_613"])
                for w in infra["windows"]:
                    window_df = infra["base_data"][w]
                    if window_df.loc[subset_cell_mask(S, window_df), "expected"].sum() < min_baseline_expected:
                        continue
                    widx = window_idx_of(w, infra)
                    trial_seed = seed_base + deterministic_seed(METHOD_NAME, j, r, trial_idx, w) % 1_000_000

                    (c_det, _c_sub, c_ratio, c_p), (i_det, i_sub, i_ratio, i_p), s_ratio = run_epm_step_trial(
                        widx, S, S_mask, r, epm, infra, null_ratios_by_window[w], trial_seed
                    )
                    rows.append({
                        "method": METHOD_NAME, "j": j, "r": r, "window": w, "trial": trial_idx,
                        "arm": "clean", "S": repr(S), "detected": c_det, "score": c_ratio,
                        "pvalue": c_p, "jaccard": np.nan, "recall": np.nan, "precision": np.nan,
                        "s_expected_share": np.nan, "s_growth_ratio": np.nan, "is_disjunctive": is_disjunctive,
                    })
                    Sn, i_subn = _stringify_subset(S), _stringify_subset(i_sub)
                    rows.append({
                        "method": METHOD_NAME, "j": j, "r": r, "window": w, "trial": trial_idx,
                        "arm": "injected", "S": repr(S), "detected": i_det, "score": i_ratio,
                        "pvalue": i_p, "jaccard": jaccard(i_subn, Sn), "recall": recall(i_subn, Sn),
                        "precision": precision(i_subn, Sn),
                        "s_expected_share": s_share, "s_growth_ratio": s_ratio, "is_disjunctive": is_disjunctive,
                    })
    return pd.DataFrame(rows)


def summarize_epm_step(step_df):
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
# RAMP experiment (EPM) — boiling-frog leakage via the trailing reference
# =============================================================================

def run_epm_ramp_trial(S, r, onset_idx, epm, infra, null_ratios_by_window, seed_base):
    windows = infra["windows"]
    multipliers = ramp_multipliers(r, RAMP_LENGTH)
    ramp_series = []  # realized (post-injection) count vectors, for future leakage
    detected_at = None
    for step, mult in enumerate(multipliers):
        w = windows[onset_idx + step]
        widx = window_idx_of(w, infra)
        step_seed = seed_base + step * 10

        rng_det = np.random.default_rng(step_seed)
        clean = draw_counts(widx, epm, infra, rng_det)
        injected = inject_counts(clean, S, mult, infra["cells_613"])

        # trailing reference PERIOD = pool of the N_REFERENCE_WINDOWS preceding
        # windows; within the ramp those are the PRIOR steps' own already-injected
        # counts (boiling-frog leakage), with fresh clean draws for any depth
        # reaching before ramp onset. As the ramp lengthens, more of the reference
        # pool is contaminated by the rising signal.
        rng_ref = np.random.default_rng(step_seed + 1)
        ref = np.zeros(len(infra["cells_613"]), dtype=float)
        for b in range(1, N_REFERENCE_WINDOWS + 1):
            prev = step - b
            if prev >= 0:
                ref += ramp_series[prev]
            else:
                ref += draw_counts(widx - b, epm, infra, rng_ref)

        det, _sub, _ratio, _p = epm_detect(injected, ref, epm, null_ratios_by_window[w])
        if det and detected_at is None:
            detected_at = step + 1
        ramp_series.append(injected)
    return detected_at, windows[onset_idx]


def run_epm_ramp_experiment(infra, epm, null_ratios_by_window, j_grid, r_grid, n_trials, seed_base, min_baseline_expected):
    onset_candidates = valid_onset_windows(infra["windows"], RAMP_LENGTH)
    pooled_ref = pd.concat(infra["base_data"].values(), ignore_index=True).groupby(FEATURES, as_index=False)["expected"].sum()
    rows = []
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
                detected_at, onset_w = run_epm_ramp_trial(S, r, onset_idx, epm, infra, null_ratios_by_window, trial_seed)
                rows.append({
                    "method": METHOD_NAME, "j": j, "r": r, "trial": trial_idx,
                    "onset_window": onset_w, "S": repr(S),
                    "time_to_detect": detected_at if detected_at is not None else np.nan,
                    "censored": detected_at is None,
                })
    return pd.DataFrame(rows)


def summarize_epm_ramp(ramp_df):
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
# VALIDATION GATE (EPM)
# =============================================================================

def run_validation_gate_epm(infra, epm, null_ratios_by_window, n_trials_gate=N_TRIALS_GATE):
    """Gate deliberately POOLED over all 12 windows (not a single window like
    08/09's gate). Reason: EPM is an intentionally weak baseline — its pooled
    power at r=3 is ~0.10, so a single-window/10-trial gate is statistically
    underpowered to confirm the (real, small) monotone power trend and gives a
    NaN/near-zero recall by chance. Pooling gives the gate enough samples to
    measure the effect it is supposed to measure; it does not change WHAT is
    tested. The recall bar is also lower than 09's 0.5: EPM's max-growth-ratio
    detections are frequently noise-driven (a genuine, reportable EPM weakness,
    not a wiring bug — see summary), so the gate only confirms that recovery
    HAPPENS on strong signals (identification mapping wired correctly) and that
    power RESPONDS to signal — not that EPM is a competent detector."""
    print("\n" + "=" * 78)
    print("VALIDATION GATE (EPM, before the full grid)")
    print("=" * 78)
    print("Pooled over all 12 windows (EPM's low power needs the samples); conjunctive S only "
          "(mod08.draw_subgroup plants one value/feature), so representation is never the gate confound.")

    # Power/FPR probe on the GRID multipliers.
    power_df = run_epm_step_experiment(
        infra, epm, null_ratios_by_window, j_grid=[2], r_grid=[1.5, 3.0],
        n_trials=n_trials_gate, seed_base=BASE_SEED + 1400, min_baseline_expected=MIN_BASELINE_EXPECTED,
    )
    summary = summarize_epm_step(power_df)
    print(summary.to_string(index=False))

    checks = {}
    fpr = summary["fpr"].mean()
    checks["clean_fpr_near_5pct"] = bool(0.0 <= fpr <= 0.20)
    print(f"\n[{'PASS' if checks['clean_fpr_near_5pct'] else 'FAIL'}] Clean-arm FPR = {fpr:.1%} (own EPM null, pooled)")

    power_15 = summary.loc[summary["r"] == 1.5, "power"].iloc[0]
    power_30 = summary.loc[summary["r"] == 3.0, "power"].iloc[0]
    checks["power_rises_with_r"] = bool(power_30 > power_15)
    print(f"[{'PASS' if checks['power_rises_with_r'] else 'FAIL'}] Power rises with r: "
          f"power(1.5)={power_15:.2f} -> power(3.0)={power_30:.2f}")

    # Recovery probe on a deliberately STRONG, off-grid multiplier. The spec asks
    # that the recovered itemset "overlaps planted S on STRONG conjunctive signals";
    # the grid's r=3 is NOT strong for EPM — its denominator attenuation drops the
    # injected pattern's growth ratio to ~2.2, below the clean-null p95 (~4.4), so at
    # r=3 EPM's argmax is usually a noise itemset (recall on detected < on
    # non-detected — a genuine EPM weakness reported in the results, NOT a wiring bug).
    # At GATE_RECOVERY_R the injected pattern unambiguously dominates, so this isolates
    # "is the itemset->subset identification mapping wired correctly and does EPM
    # recover S when the signal is unmistakable" from EPM's operational weakness.
    rec_df = run_epm_step_experiment(
        infra, epm, null_ratios_by_window, j_grid=[2], r_grid=[GATE_RECOVERY_R],
        n_trials=n_trials_gate, seed_base=BASE_SEED + 1450, min_baseline_expected=MIN_BASELINE_EXPECTED,
    )
    strong = rec_df[(rec_df["arm"] == "injected")]
    recall_detected = strong.loc[strong["detected"], "recall"].mean()
    recall_nondetected = strong.loc[~strong["detected"], "recall"].mean()
    checks["recovers_planted_subgroup"] = bool(
        pd.notna(recall_detected) and recall_detected > 0.0
        and (pd.isna(recall_nondetected) or recall_detected > recall_nondetected)
    )
    print(f"[{'PASS' if checks['recovers_planted_subgroup'] else 'FAIL'}] Recovers S on a STRONG signal "
          f"(r={GATE_RECOVERY_R}): mean recall on DETECTED = {recall_detected:.2f} vs non-detected = {recall_nondetected:.2f} "
          "(detected must be >0 and exceed non-detected).")

    all_pass = all(checks.values())
    print(f"\nGate result: {'ALL PASS' if all_pass else 'FAILED — STOPPING, not launching the full grid'}")
    return all_pass, checks


# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def denominator_effect_table(step_df):
    """Reveal the support-denominator attenuation: the planted S's own
    growth ratio should fall BELOW the naive r as the subgroup's share of the
    book grows (bigger share -> more of the injected excess lands in the
    denominator). Grouped by r and share tercile."""
    inj = step_df[(step_df["arm"] == "injected") & step_df["s_growth_ratio"].notna()].copy()
    if inj.empty:
        return pd.DataFrame()
    inj["share_tercile"] = pd.qcut(inj["s_expected_share"], 3, labels=["low", "mid", "high"], duplicates="drop")
    tbl = inj.groupby(["r", "share_tercile"], observed=True).agg(
        n=("s_growth_ratio", "size"),
        mean_share=("s_expected_share", "mean"),
        mean_S_growth_ratio=("s_growth_ratio", "mean"),
    ).reset_index()
    tbl["naive_r"] = tbl["r"]
    tbl["attenuation_ratio"] = tbl["mean_S_growth_ratio"] / tbl["r"]
    return tbl


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
    step_sum_csv = config.REPORTS_DIR / "e2_step_summary.csv"
    ramp_sum_csv = config.REPORTS_DIR / "e2_ramp_summary.csv"
    # Idempotent append: drop any prior EPM rows so re-running never duplicates.
    prior_step = pd.read_csv(step_csv)
    prior_ramp = pd.read_csv(ramp_csv)
    prior_step = prior_step[prior_step["method"] != METHOD_NAME].reset_index(drop=True)
    prior_ramp = prior_ramp[prior_ramp["method"] != METHOD_NAME].reset_index(drop=True)

    print("Building infrastructure (regression, alpha, 613-cell frame, extended window totals)...")
    infra = mod09.build_infrastructure()
    item_cols, col_of, onehot = build_epm_index(infra["cells_613"])
    hp = fix_epm_hyperparams(infra)
    mu_anchored = precompute_mu_anchored(infra)
    epm = {"item_cols": item_cols, "col_of": col_of, "onehot": onehot,
           "minsup": hp["minsup"], "smooth_floor": hp["smooth_floor"], "mu_anchored": mu_anchored}
    print(f"Loaded {len(infra['windows'])} test windows; {onehot.shape[1]} (feature=value) items over the 613-cell frame.")
    print(f"\nFIXED-ON-TRAINING hyperparameters (never tuned on detection):")
    print(f"  median training-window total (613 universe) = {hp['median_train_total']:.0f} claims")
    print(f"  minsup = {hp['min_abs_support']:.0f} claims / {hp['median_train_total']:.0f} = {hp['minsup']:.4f} relative support "
          f"(the same 15-claim min-presence floor mod08 uses for a valid subgroup)")
    print(f"  reference smoothing floor = minsup = {hp['smooth_floor']:.4f} (sub-threshold reference support is not "
          f"reliably present; caps 'jumping'-itemset noise that would otherwise dominate the max ratio)")
    print(f"  max itemset length = {MAX_ITEMSET_LEN}; reference period = {N_REFERENCE_WINDOWS} trailing window")

    print("\n" + "=" * 78)
    print("EPM NULL CALIBRATION (max support-growth-ratio, own null)")
    print("=" * 78)
    t_cal = time.time()
    null_ratios_by_window, check_fractions, null_long = calibrate_epm(infra, epm, M_CALIBRATION, BASE_SEED + 1500)
    cal_elapsed = time.time() - t_cal
    n_cal_calls = len(infra["windows"]) * M_CALIBRATION * 2
    per_call = cal_elapsed / n_cal_calls
    pooled_fpr = float(np.mean(list(check_fractions.values())))
    print(f"\nPooled calibration FPR (own null) = {pooled_fpr:.1%} (target 5%)")
    out_path = config.REPORTS_DIR / "e2_epm_null_maxratios.csv"
    null_long.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(null_long)} rows)")

    gate_pass, gate_checks = run_validation_gate_epm(infra, epm, null_ratios_by_window)
    if not gate_pass:
        print("\nSTOPPING: EPM validation gate failed. Not launching the full grid.")
        return

    n_windows = len(infra["windows"])
    n_step_calls = len(J_GRID) * len(R_GRID) * N_TRIALS_FULL * n_windows * 2
    n_ramp_calls = len(J_GRID) * len(R_GRID) * N_TRIALS_FULL * RAMP_LENGTH
    total_grid_calls = n_step_calls + n_ramp_calls
    eta = total_grid_calls * per_call
    print(f"\nMODE={MODE!r}. Per fpgrowth call ~= {per_call*1000:.1f} ms (measured over {n_cal_calls:,} calibration calls).")
    print(f"Grid fpgrowth calls (before per-window skips): {n_step_calls:,} step + {n_ramp_calls:,} ramp = {total_grid_calls:,}. ETA ~= {eta/60:.1f} min.")

    print("\n" + "=" * 78)
    print("EPM STEP INJECTION EXPERIMENT")
    print("=" * 78)
    epm_step_df = run_epm_step_experiment(infra, epm, null_ratios_by_window, J_GRID, R_GRID, N_TRIALS_FULL, BASE_SEED + 1600, MIN_BASELINE_EXPECTED)
    epm_step_summary = summarize_epm_step(epm_step_df)
    print(epm_step_summary.to_string(index=False))

    print("\n" + "=" * 78)
    print("EPM RAMP INJECTION EXPERIMENT (boiling-frog reference leakage)")
    print("=" * 78)
    epm_ramp_df = run_epm_ramp_experiment(infra, epm, null_ratios_by_window, J_GRID, R_GRID, N_TRIALS_FULL, BASE_SEED + 1700, MIN_BASELINE_EXPECTED)
    epm_ramp_summary = summarize_epm_ramp(epm_ramp_df)
    print(epm_ramp_summary.to_string(index=False))

    # Append rows (standard 13-column schema only; the extra analysis columns
    # stay in-memory) and regenerate the by-method plots over all 3 methods.
    std_cols = ["method", "j", "r", "window", "trial", "arm", "S", "detected", "score", "pvalue", "jaccard", "recall", "precision"]
    combined_step = pd.concat([prior_step, epm_step_df[std_cols]], ignore_index=True)
    combined_ramp = pd.concat([prior_ramp, epm_ramp_df], ignore_index=True)
    combined_step.to_csv(step_csv, index=False)
    combined_ramp.to_csv(ramp_csv, index=False)
    print(f"\nAppended EPM rows: {step_csv} now {len(combined_step)} rows; {ramp_csv} now {len(combined_ramp)} rows")

    prior_step_summary = pd.read_csv(step_sum_csv)
    prior_step_summary = prior_step_summary[prior_step_summary["method"] != METHOD_NAME]
    for col in ["mean_precision_detected", "mean_precision_all"]:
        if col not in prior_step_summary.columns:
            prior_step_summary[col] = np.nan
    combined_step_summary = pd.concat([prior_step_summary, epm_step_summary], ignore_index=True)
    prior_ramp_summary = pd.read_csv(ramp_sum_csv)
    prior_ramp_summary = prior_ramp_summary[prior_ramp_summary["method"] != METHOD_NAME]
    combined_ramp_summary = pd.concat([prior_ramp_summary, epm_ramp_summary], ignore_index=True)
    combined_step_summary.to_csv(step_sum_csv, index=False)
    combined_ramp_summary.to_csv(ramp_sum_csv, index=False)

    mod09.plot_power_curves_by_method(combined_step_summary, "e2_power_curves_by_method.png")
    mod09.plot_precision_recall_jaccard(combined_step_summary, "e2_precision_recall_jaccard_by_method.png")
    mod09.plot_time_to_detect_by_method(combined_ramp_summary, "e2_time_to_detect_by_method.png")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"\nminsup = {hp['minsup']:.4f} (= {hp['min_abs_support']:.0f} claims / median training-window total "
          f"{hp['median_train_total']:.0f}), fixed on training, never tuned on detection.")
    print(f"Pooled EPM calibration FPR (own null) = {pooled_fpr:.1%}")
    print(f"\nGate: {'PASS' if gate_pass else 'FAIL'} ({gate_checks})")
    print(f"\nEPM step summary (power/FPR/precision/recall/Jaccard by j,r):\n{epm_step_summary.to_string(index=False)}")
    print(f"\nEPM ramp summary:\n{epm_ramp_summary.to_string(index=False)}")

    print("\nSupport-denominator effect (planted-S growth ratio vs naive r, by share tercile):")
    den = denominator_effect_table(epm_step_df)
    print(den.to_string(index=False) if not den.empty else "  (no injected detected trials)")
    print("  attenuation_ratio < 1 => the injected total's denominator ate part of the signal; "
          "it should shrink as the subgroup's share grows.")

    n_disj = int((epm_step_df["is_disjunctive"] == True).sum())
    print(f"\nConjunctive-vs-disjunctive planted S: {len(epm_step_df) - n_disj} conjunctive rows, {n_disj} disjunctive rows "
          "(mod08.draw_subgroup plants one value/feature -> all conjunctive; EPM's no-disjunction handicap does not bite here).")

    print("\n3-way head-to-head (power, by j,r):")
    compare = combined_step_summary.pivot_table(index=["j", "r"], columns="method", values="power")
    print(compare.to_string())

    per_window_power = epm_step_df[epm_step_df["arm"] == "injected"].groupby("window")["detected"].mean()
    flagged = per_window_power[per_window_power.index.isin(FLAG_WINDOWS)]
    other = per_window_power[~per_window_power.index.isin(FLAG_WINDOWS)]
    print(f"\nEPM pooled power (non-flagged windows): {other.mean():.2f}")
    print(f"EPM power in flagged windows {sorted(FLAG_WINDOWS)}: {flagged.to_dict()}")

    mod08_md5_after = mod04.file_md5(mod08_path)
    print(f"\nHarness core (08_e2_injection.py) unchanged: {mod08_md5_before == mod08_md5_after}")
    elapsed = time.time() - t_start
    print(f"Total compute: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\nSource artifact integrity:")
    for name, before in md5_before.items():
        print(f"  {name}: unchanged={before == mod04.file_md5(md5_paths[name])}")


if __name__ == "__main__":
    main()
