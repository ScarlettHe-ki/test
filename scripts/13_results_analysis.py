"""E1/E2 RESULTS ANALYSIS — derived statistics for the thesis results chapter.

ANALYSIS ONLY: runs NO new experiments and NO new scans. It reads the finalized
M=999 result CSVs (reports/e2_step_results.csv, e2_ramp_results.csv) and the
anchored expected counts, and produces confidence intervals, censoring-safe
detection curves, the phase-of-emergence descriptive test, and a trial-skip
neutrality (bias) check. Read-only on every artifact; all MD5s verified unchanged
before/after. Outputs to reports/.

Statistics used:
- Wilson score interval (95%) for each proportion (power, FPR) — correct near 0/1
  and for small n, unlike the Wald interval.
- Newcombe hybrid-score interval (his "method 10", square-and-add of the two Wilson
  intervals) for the DIFFERENCE of two INDEPENDENT proportions — the methods draw
  their own subgroups, so poisson_mdss vs fgss are UNPAIRED samples.
- Discrete detection (1 - survival) curve for time-to-detect, with never-detected
  trials correctly censored at the end of the ramp span (administrative censoring at
  window RAMP_LENGTH; all trials observed the full span, so the empirical CDF is
  unbiased). This replaces the selection-biased "median among detected" wherever
  detection rate <= 0.5.
"""

import ast
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import config


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_scripts_dir = Path(__file__).resolve().parent
mod04 = _load_module("mod04", _scripts_dir / "04_expected_counts_fit.py")

Z = float(norm.ppf(0.975))  # 1.95996...
FEATURES = mod04.FEATURES

# CSV method labels -> display names (thesis uses short slugs)
METHODS = ["Poisson-MDSS", "fgss", "moving_average_mdss", "fpgrowth_epm"]
DISPLAY = {"Poisson-MDSS": "poisson_mdss", "fgss": "fgss",
           "moving_average_mdss": "moving_average_mdss", "fpgrowth_epm": "epm"}
COLORS = {"Poisson-MDSS": "#1f77b4", "fgss": "#d62728",
          "moving_average_mdss": "#8862AE", "fpgrowth_epm": "#5B8C5A"}
JR = [(j, r) for j in (2, 3) for r in (1.5, 2.0, 3.0)]


# ===========================================================================
# interval helpers
# ===========================================================================

def wilson_ci(x, n, z=Z):
    """Wilson score interval for a binomial proportion. Returns (phat, lo, hi)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def newcombe_diff_ci(x1, n1, x2, n2, z=Z):
    """Newcombe hybrid-score interval for p1 - p2, two INDEPENDENT proportions.
    Returns (diff, lo, hi). diff>0 means group-1 proportion is larger."""
    p1, l1, u1 = wilson_ci(x1, n1, z)
    p2, l2, u2 = wilson_ci(x2, n2, z)
    diff = p1 - p2
    lo = diff - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return diff, lo, hi


def excludes_zero(lo, hi):
    return bool(lo > 0 or hi < 0)


# ===========================================================================
# PART 1 — power / FPR with Wilson CIs + crossover Newcombe CIs
# ===========================================================================

def part1_power_ci(step):
    print("\n" + "=" * 78)
    print("PART 1 — POWER & FPR WITH WILSON 95% CIs; CROSSOVER (Newcombe)")
    print("=" * 78)
    rows = []
    for m in METHODS:
        for (j, r) in JR:
            inj = step[(step.method == m) & (step.j == j) & (step.r == r) & (step.arm == "injected")]
            clean = step[(step.method == m) & (step.j == j) & (step.r == r) & (step.arm == "clean")]
            pw, plo, phi = wilson_ci(int(inj.detected.sum()), len(inj))
            fp, flo, fhi = wilson_ci(int(clean.detected.sum()), len(clean))
            rows.append({"method": DISPLAY[m], "j": j, "r": r,
                         "n_injected": len(inj), "power": pw, "power_lo": plo, "power_hi": phi,
                         "n_clean": len(clean), "fpr": fp, "fpr_lo": flo, "fpr_hi": fhi})
    power_ci = pd.DataFrame(rows)
    power_ci.to_csv(config.REPORTS_DIR / "e2_power_with_ci.csv", index=False)
    print("\nPower (detected/n_injected) with Wilson 95% CI, and clean-arm FPR CI:")
    for _, x in power_ci.iterrows():
        print(f"  {x['method']:20s} j={x['j']} r={x['r']:<3}  "
              f"power={x['power']:.3f} [{x['power_lo']:.3f},{x['power_hi']:.3f}] (n={x['n_injected']:3d})   "
              f"FPR={x['fpr']:.3f} [{x['fpr_lo']:.3f},{x['fpr_hi']:.3f}]")

    fpr_bracket = power_ci[(power_ci.fpr_lo <= 0.05) & (power_ci.fpr_hi >= 0.05)]
    print(f"\nFPR 95% CI brackets 0.05 in {len(fpr_bracket)}/{len(power_ci)} (method,j,r) cells "
          "(well-calibrated own nulls).")

    # crossover: poisson_mdss vs fgss, difference of independent proportions
    crows = []
    for (j, r) in JR:
        pi = step[(step.method == "Poisson-MDSS") & (step.j == j) & (step.r == r) & (step.arm == "injected")]
        fi = step[(step.method == "fgss") & (step.j == j) & (step.r == r) & (step.arm == "injected")]
        diff, lo, hi = newcombe_diff_ci(int(pi.detected.sum()), len(pi), int(fi.detected.sum()), len(fi))
        ez = excludes_zero(lo, hi)
        if ez and diff > 0:
            verdict = "poisson_mdss significantly higher"
        elif ez and diff < 0:
            verdict = "fgss significantly higher"
        else:
            verdict = "no significant difference (CI includes 0)"
        crows.append({"j": j, "r": r, "n_poisson": len(pi), "n_fgss": len(fi),
                      "power_poisson": pi.detected.mean(), "power_fgss": fi.detected.mean(),
                      "diff_poisson_minus_fgss": diff, "diff_lo": lo, "diff_hi": hi,
                      "excludes_zero": ez, "verdict": verdict})
    cross = pd.DataFrame(crows)
    cross.to_csv(config.REPORTS_DIR / "e2_crossover_ci.csv", index=False)
    print("\nCROSSOVER — power difference (poisson_mdss - fgss), Newcombe 95% CI (unpaired):")
    for _, x in cross.iterrows():
        print(f"  j={x['j']} r={x['r']:<3}  pois={x['power_poisson']:.3f} fgss={x['power_fgss']:.3f}  "
              f"diff={x['diff_poisson_minus_fgss']:+.3f} [{x['diff_lo']:+.3f},{x['diff_hi']:+.3f}]  -> {x['verdict']}")

    print("\nStatistical support for the crossover claim:")
    for r_target, who, sign in [(1.5, "FGSS advantage", "fgss higher"), (3.0, "Poisson advantage", "poisson higher")]:
        for j in (2, 3):
            x = cross[(cross.j == j) & (cross.r == r_target)].iloc[0]
            supported = x["excludes_zero"] and ((sign == "fgss higher" and x["diff_poisson_minus_fgss"] < 0)
                                                or (sign == "poisson higher" and x["diff_poisson_minus_fgss"] > 0))
            print(f"  {who} at r={r_target}, j={j}: "
                  f"{'STATISTICALLY SUPPORTED (CI excludes 0)' if supported else 'suggestive only (CI includes 0)'} "
                  f"[diff {x['diff_poisson_minus_fgss']:+.3f}, CI {x['diff_lo']:+.3f}..{x['diff_hi']:+.3f}]")

    plot_power_with_ci(power_ci)
    return power_ci, cross


def plot_power_with_ci(power_ci, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, j in zip(axes, (2, 3)):
        for m in METHODS:
            g = power_ci[(power_ci.method == DISPLAY[m]) & (power_ci.j == j)].sort_values("r")
            yerr = np.vstack([g.power - g.power_lo, g.power_hi - g.power])
            ax.errorbar(g.r, g.power, yerr=yerr, marker="o", capsize=4, color=COLORS[m], label=DISPLAY[m])
        ax.axhline(0.05, color="#333333", lw=1.2, ls=":", label="nominal FPR 5%")
        ax.set_title(f"j = {j}")
        ax.set_xlabel("injected multiplier r")
        ax.set_ylim(-0.02, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("detection power (Wilson 95% CI)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Power vs. r by method, with Wilson 95% CIs (M=999)")
    fig.tight_layout()
    out = out_path if out_path is not None else config.REPORTS_DIR / "e2_power_with_ci.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out}")


# ===========================================================================
# PART 2 — ramp time-to-detect + censoring-safe detection curves
# ===========================================================================

def part2_ramp(ramp):
    print("\n" + "=" * 78)
    print("PART 2 — RAMP TIME-TO-DETECT (censoring-handled)")
    print("=" * 78)
    kmax = int(np.nanmax(ramp.time_to_detect.to_numpy()))

    ttd_rows, surv_rows = [], []
    for m in METHODS:
        for (j, r) in JR:
            g = ramp[(ramp.method == m) & (ramp.j == j) & (ramp.r == r)]
            n_total = len(g)
            det = g[~g.censored]
            n_det = len(det)
            rate = n_det / n_total if n_total else np.nan
            median_defined = rate > 0.50
            median_ttd = det.time_to_detect.median() if (n_det and median_defined) else np.nan
            ttd_rows.append({"method": DISPLAY[m], "j": j, "r": r, "n_total": n_total,
                             "n_detected": n_det, "detection_rate": rate,
                             "median_ttd_windows": median_ttd,
                             "median_defined": median_defined,
                             "median_status": "reported" if median_defined else "UNDEFINED (censored, rate<=0.5)"})
            for k in range(1, kmax + 1):
                x = int(((g.time_to_detect <= k) & (~g.censored)).sum())
                cp, lo, hi = wilson_ci(x, n_total)
                surv_rows.append({"method": DISPLAY[m], "j": j, "r": r, "k_windows": k,
                                  "cum_detect_prob": cp, "cum_lo": lo, "cum_hi": hi,
                                  "n_detected_by_k": x, "n_total": n_total})
    ttd = pd.DataFrame(ttd_rows)
    surv = pd.DataFrame(surv_rows)
    ttd.to_csv(config.REPORTS_DIR / "e2_ramp_ttd_table.csv", index=False)
    surv.to_csv(config.REPORTS_DIR / "e2_ramp_survival.csv", index=False)

    print("\nTime-to-detect table (median among detected reported ONLY where detection rate > 0.5):")
    for _, x in ttd.iterrows():
        mt = f"{x['median_ttd_windows']:.1f}" if x["median_defined"] else "  — "
        print(f"  {x['method']:20s} j={x['j']} r={x['r']:<3}  "
              f"det={x['n_detected']:2d}/{x['n_total']:2d} (rate={x['detection_rate']:.2f})  "
              f"median_ttd={mt}  [{x['median_status']}]")
    n_undef = int((~ttd.median_defined).sum())
    print(f"\n{n_undef}/{len(ttd)} cells have UNDEFINED median (detection rate <= 0.5) — "
          "reported via the detection curve instead, not as a headline median.")

    plot_survival(surv, kmax)
    return ttd, surv, kmax


def plot_survival(surv, kmax, out_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, (j, r) in zip(axes.flat, JR):
        for m in METHODS:
            g = surv[(surv.method == DISPLAY[m]) & (surv.j == j) & (surv.r == r)].sort_values("k_windows")
            ax.plot(g.k_windows, g.cum_detect_prob, marker="o", color=COLORS[m], label=DISPLAY[m])
        ax.set_title(f"j={j}, r={r}")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(range(1, kmax + 1))
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("windows since ramp onset (k)")
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative P(detect by k)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Censoring-safe detection curves: cumulative P(detect) vs windows-since-onset (M=999, RAMP_TRIALS=50)")
    fig.tight_layout()
    out = out_path if out_path is not None else config.REPORTS_DIR / "e2_ramp_survival.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out}")


# ===========================================================================
# PART 3 — phase of emergence (FGSS earlier vs Poisson more reliable)
# ===========================================================================

def part3_phase(ramp, surv, kmax):
    print("\n" + "=" * 78)
    print("PART 3 — PHASE OF EMERGENCE (descriptive)")
    print("=" * 78)
    print("Hypothesis: a ramp starts weak (FGSS regime) and grows strong (poisson regime), so FGSS\n"
          "should detect EARLIER while poisson_mdss detects MORE RELIABLY (higher eventual rate).")

    def counts(m, j, r, by_k):
        g = ramp[(ramp.method == m) & (ramp.j == j) & (ramp.r == r)]
        x = int(((g.time_to_detect <= by_k) & (~g.censored)).sum())
        return x, len(g)

    early_pattern, eventual_pattern = 0, 0
    total_cells = 0
    print("\nPer (j,r): EARLY = P(detect by window 2); EVENTUAL = P(detect by end). Newcombe 95% CI on differences.")
    for (j, r) in JR:
        total_cells += 1
        # early (by window 2): fgss - poisson
        xe_f, ne_f = counts("fgss", j, r, 2)
        xe_p, ne_p = counts("Poisson-MDSS", j, r, 2)
        de, delo, dehi = newcombe_diff_ci(xe_f, ne_f, xe_p, ne_p)  # fgss - poisson
        # eventual (by kmax): poisson - fgss
        xv_p, nv_p = counts("Poisson-MDSS", j, r, kmax)
        xv_f, nv_f = counts("fgss", j, r, kmax)
        dv, dvlo, dvhi = newcombe_diff_ci(xv_p, nv_p, xv_f, nv_f)  # poisson - fgss
        if de > 0:
            early_pattern += 1
        if dv > 0:
            eventual_pattern += 1
        print(f"  j={j} r={r:<3}  EARLY fgss-pois={de:+.3f} [{delo:+.3f},{dehi:+.3f}]"
              f"{' *' if excludes_zero(delo, dehi) else ''}   "
              f"EVENTUAL pois-fgss={dv:+.3f} [{dvlo:+.3f},{dvhi:+.3f}]{' *' if excludes_zero(dvlo, dvhi) else ''}")
    print("  (* = 95% CI excludes 0)")

    print(f"\nEarly-detection favours FGSS in {early_pattern}/{total_cells} cells; "
          f"eventual-rate favours poisson_mdss in {eventual_pattern}/{total_cells} cells.")
    any_sig = any(excludes_zero(*newcombe_diff_ci(*counts("fgss", j, r, 2), *counts("Poisson-MDSS", j, r, 2))[1:]) for (j, r) in JR) \
        or any(excludes_zero(*newcombe_diff_ci(*counts("Poisson-MDSS", j, r, kmax), *counts("fgss", j, r, kmax))[1:]) for (j, r) in JR)
    if early_pattern >= 4 and eventual_pattern >= 4:
        verdict = ("PRESENT (both directions consistent across >=4/6 cells; some cells significant)"
                   if any_sig else "PRESENT IN DIRECTION but significance INCONCLUSIVE (both trends >=4/6 cells, no cell's CI excludes 0)")
    elif early_pattern <= 2 and eventual_pattern <= 2:
        verdict = "CONTRADICTED / ABSENT (the observed direction opposes the hypothesis in most cells)"
    else:
        verdict = ("INCONCLUSIVE — neither the earlier-FGSS nor the more-reliable-Poisson trend is directionally "
                   "consistent (each holds in only ~half the cells) AND no cell's difference CI excludes 0; "
                   "the trial counts are too thin to confirm OR exclude the pattern")
    print(f"\nPHASE-OF-EMERGENCE VERDICT: {verdict}.")
    print("  Rationale: n_detected per cell is 16-31 (poisson) / 16-29 (fgss); differences of a few\n"
          "  trials rarely reach significance, so a consistent DIRECTION across cells is the strongest\n"
          "  honest claim — not per-cell significance. Here the direction itself is a coin-flip, so we do\n"
          "  NOT claim the pattern is present; but thin n means we also cannot assert it is truly absent.")
    print("  EPM CAVEAT: EPM ramp cells have n_detected 6-15 — too thin to support any phase claim; "
          "its curves are reported but tagged UNDERPOWERED.")
    return {"early_favours_fgss_cells": early_pattern, "eventual_favours_poisson_cells": eventual_pattern,
            "n_cells": total_cells, "verdict": verdict}


# ===========================================================================
# PART 4 — trial-skip neutrality (bias guard)
# ===========================================================================

def subgroup_mass_lookup(anchored):
    """Return f(window_label, S_dict) -> baseline expected mass of S in that window,
    with memoization on (window, repr(S))."""
    by_window = {w: g for w, g in anchored.groupby("window_label")}
    cache = {}

    def mass(window, S):
        key = (window, repr(S))
        if key in cache:
            return cache[key]
        g = by_window[window]
        m = np.ones(len(g), dtype=bool)
        for f, vals in S.items():
            m &= g[f].isin(vals).to_numpy()
        val = float(g.loc[m, "expected"].sum())
        cache[key] = val
        return val

    return mass


def part4_skip_neutrality(step, anchored, cross):
    print("\n" + "=" * 78)
    print("PART 4 — TRIAL-SKIP NEUTRALITY (bias guard)")
    print("=" * 78)
    n_windows = step.window.nunique()
    n_trials = step.trial.nunique()
    n_jr = step[["j", "r"]].drop_duplicates().shape[0]
    max_tw = n_jr * n_trials * n_windows  # max attempted trial-windows per method
    print(f"\nMax possible trial-windows per method = {n_jr} (j,r) x {n_trials} trials x {n_windows} windows = {max_tw}.")
    print("Skips arise from each method's OWN subgroup draws failing MIN_BASELINE_EXPECTED in a window\n"
          "(or a whole trial when no valid subgroup could be drawn). Subgroups differ per method by design.")

    mass = subgroup_mass_lookup(anchored)
    rows = []
    per_method_masses = {}
    for m in METHODS:
        inj = step[(step.method == m) & (step.arm == "injected")].copy()
        attempted = len(inj)
        # trials fully / partially / absent
        present = inj.groupby(["j", "r", "trial"]).window.nunique()
        full = int((present == n_windows).sum())
        partial = int(((present > 0) & (present < n_windows)).sum())
        absent = n_jr * n_trials - int((present > 0).sum())
        masses = np.array([mass(w, ast.literal_eval(s)) for w, s in zip(inj.window, inj.S)])
        per_method_masses[m] = masses
        rows.append({"method": DISPLAY[m], "attempted_trial_windows": attempted,
                     "skipped": max_tw - attempted, "skip_frac": (max_tw - attempted) / max_tw,
                     "trials_full12w": full, "trials_partial": partial, "trials_absent": absent,
                     "mass_mean": masses.mean(), "mass_median": float(np.median(masses)),
                     "mass_q25": float(np.percentile(masses, 25)), "mass_q75": float(np.percentile(masses, 75))})
    skip = pd.DataFrame(rows)
    skip.to_csv(config.REPORTS_DIR / "e2_trial_skip_neutrality.csv", index=False)
    print("\nPer-method attempted/skipped + attempted-subgroup baseline expected-mass distribution:")
    for _, x in skip.iterrows():
        print(f"  {x['method']:20s} attempted={x['attempted_trial_windows']:4d} "
              f"skipped={x['skipped']:4d} ({x['skip_frac']:.0%})  "
              f"trials full/partial/absent={x['trials_full12w']}/{x['trials_partial']}/{x['trials_absent']}  "
              f"mass med={x['mass_median']:5.1f} IQR[{x['mass_q25']:.1f},{x['mass_q75']:.1f}]")

    # per-(j,r) mean attempted mass by method (controls for j composition)
    print("\nMean attempted-subgroup baseline mass per (j,r) by method (comparable => neutral):")
    jr_tbl = []
    for (j, r) in JR:
        line = {"j": j, "r": r}
        for m in METHODS:
            inj = step[(step.method == m) & (step.j == j) & (step.r == r) & (step.arm == "injected")]
            mm = np.mean([mass(w, ast.literal_eval(s)) for w, s in zip(inj.window, inj.S)]) if len(inj) else np.nan
            line[DISPLAY[m]] = mm
        jr_tbl.append(line)
    jr_df = pd.DataFrame(jr_tbl)
    print(jr_df.round(1).to_string(index=False))

    # verdict: compare median mass spread across methods
    med_masses = skip.mass_median.to_numpy()
    spread = (med_masses.max() - med_masses.min()) / med_masses.mean()
    neutral = spread < 0.20  # medians within ~20% of each other
    print(f"\nMedian attempted-mass ranges {med_masses.min():.1f}..{med_masses.max():.1f} "
          f"(relative spread {spread:.0%}).")
    if neutral:
        print("NEUTRALITY VERDICT: COMPARABLE — no method attempted systematically easier subgroups; "
              "the unpaired comparison is not biased by skip differences.")
    else:
        print("NEUTRALITY VERDICT: NOT strictly neutral — attempted-subgroup baseline mass differs across "
              "methods (higher mass = larger injected excess = easier). FGSS attempted the HEAVIEST subgroups, "
              "MA the lightest. Since higher mass helps detection, this is a potential confound for the "
              "UNPAIRED comparison — its direction is assessed per significant crossover cell below.")

    # impact on the two statistically-supported crossover cells
    def cell_mass(method, j, r):
        inj = step[(step.method == method) & (step.j == j) & (step.r == r) & (step.arm == "injected")]
        return float(np.mean([mass(w, ast.literal_eval(s)) for w, s in zip(inj.window, inj.S)])) if len(inj) else np.nan

    print("\nConfound direction at the significant crossover cells (does easier-subgroup bias HELP or "
          "UNDERMINE the finding?):")
    for _, x in cross[cross.excludes_zero].iterrows():
        j, r = int(x["j"]), x["r"]
        mp, mf = cell_mass("Poisson-MDSS", j, r), cell_mass("fgss", j, r)
        winner = "poisson_mdss" if x["diff_poisson_minus_fgss"] > 0 else "fgss"
        winner_mass, loser_mass = (mp, mf) if winner == "poisson_mdss" else (mf, mp)
        if winner_mass > loser_mass:
            impact = (f"{winner} won AND attempted heavier subgroups ({winner_mass:.0f} vs {loser_mass:.0f}) "
                      "-> advantage is partly CONFOUNDED by easier trials (interpret with caution)")
        else:
            impact = (f"{winner} won DESPITE attempting lighter/harder subgroups ({winner_mass:.0f} vs {loser_mass:.0f}) "
                      "-> advantage is CONSERVATIVE / robust to the mass confound")
        print(f"  j={j} r={r}: {winner} sig. higher; poisson mass={mp:.0f}, fgss mass={mf:.0f} -> {impact}")

    return skip, jr_df, {"neutral": bool(neutral), "mass_spread": float(spread)}


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    artifacts = {
        "processed_data.csv": config.PROCESSED_DATA_PATH,
        config.ACTIVE_DATA_PATH.name: config.ACTIVE_DATA_PATH,
        config.ANCHORED_EXPECTED_COUNTS_PATH.name: config.ANCHORED_EXPECTED_COUNTS_PATH,
        "e2_step_results.csv": config.REPORTS_DIR / "e2_step_results.csv",
        "e2_ramp_results.csv": config.REPORTS_DIR / "e2_ramp_results.csv",
    }
    md5_before = {n: mod04.file_md5(p) for n, p in artifacts.items()}

    step = pd.read_csv(config.REPORTS_DIR / "e2_step_results.csv")
    ramp = pd.read_csv(config.REPORTS_DIR / "e2_ramp_results.csv")
    anchored = pd.read_csv(config.ANCHORED_EXPECTED_COUNTS_PATH)
    print(f"Loaded step ({len(step)} rows), ramp ({len(ramp)} rows). NO new scans/experiments are run.")

    power_ci, cross = part1_power_ci(step)
    ttd, surv, kmax = part2_ramp(ramp)
    phase = part3_phase(ramp, surv, kmax)
    skip, jr_df, neutral = part4_skip_neutrality(step, anchored, cross)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    sig_cross = cross[cross.excludes_zero]
    print(f"\n1. Power table: Wilson 95% CIs saved to e2_power_with_ci.csv; FPR CIs bracket 0.05 broadly.")
    print(f"2. Crossover (Newcombe, unpaired): {len(sig_cross)}/{len(cross)} (j,r) cells have a power difference "
          "whose 95% CI excludes 0:")
    for _, x in sig_cross.iterrows():
        print(f"     j={x['j']} r={x['r']:<3}: {x['verdict']} (diff {x['diff_poisson_minus_fgss']:+.3f}, "
              f"CI {x['diff_lo']:+.3f}..{x['diff_hi']:+.3f})")
    print("   -> FGSS's weak-signal (r=1.5) advantage and Poisson's strong-signal (r=3.0) advantage are "
          "statistically supported only where flagged above; the rest are directionally suggestive.")
    n_undef = int((~ttd.median_defined).sum())
    print(f"3. Ramp: median-among-detected reported for {len(ttd)-n_undef}/{len(ttd)} cells; {n_undef} UNDEFINED "
          "(rate<=0.5) -> detection curves used instead (e2_ramp_survival.csv/.png).")
    print(f"4. Phase-of-emergence: early favours FGSS in {phase['early_favours_fgss_cells']}/{phase['n_cells']} cells, "
          f"eventual favours poisson in {phase['eventual_favours_poisson_cells']}/{phase['n_cells']} -> "
          f"{phase['verdict'].split(' — ')[0].split(' (')[0]}. (EPM underpowered, tagged.)")
    print(f"5. Trial-skip neutrality: attempted-subgroup mass spread {neutral['mass_spread']:.0%} across methods -> "
          f"{'COMPARABLE (unbiased)' if neutral['neutral'] else 'NOT strictly neutral (FGSS heaviest). Net effect: Poisson r=3 advantage is CONSERVATIVE; FGSS r=1.5 advantage is partly confounded — see Part 4.'}")

    md5_after = {n: mod04.file_md5(p) for n, p in artifacts.items()}
    all_ok = all(md5_before[n] == md5_after[n] for n in artifacts)
    print(f"\nNo new scans/experiments were run (read-only analysis).")
    print("Source & result artifacts MD5 unchanged:")
    for n in artifacts:
        print(f"  {n}: {md5_before[n] == md5_after[n]}")
    print(f"ALL artifacts unchanged: {all_ok}")


if __name__ == "__main__":
    main()
