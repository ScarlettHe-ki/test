"""E3 — real-data emerging-driver DISCOVERY scan. The thesis's actual findings.

NO injection, NO grid, NO synthetic data anywhere. This scans the REAL test-period
observed counts against the anchored expected counts and reports subgroups that are
significant against E1's M=999 per-window null. It then asks which findings PERSIST
(the emergence signature), applies the reduced-confidence caveats that determine what
is reportable, surfaces the new-level cells the scan cannot reach by construction, and
emits an exposure-check handoff table for the industry supervisor.

Imports scan() (06, scorer=None => the validated Poisson path) and pvalue() (07)
rather than reimplementing them. Read-only on every data/ file and reports/ artifact;
all MD5s verified unchanged.

RESTART COUNT (stated, and it matters): the observed scan uses N_RESTARTS = 3, matched
EXACTLY to the E1 null calibration (07's N_RESTARTS_NULL). This is essential for an
UNBIASED p-value — scanning the real data with more restarts than the null used would
find systematically higher optima and inflate significance. Matching restart counts
keeps p = P(null_max >= observed) fair.

MULTIPLE TESTING: comparing the observed score to the MAX score over the whole search
space already corrects for testing many subsets within a window (Kulldorff/Neill), so
no FDR is needed for the rank-1 subgroup per window. But we scan 12 windows, so we ALSO
report a Bonferroni threshold (0.05/12) and state which findings survive raw vs
Bonferroni — both, never silently one.
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
N_RESTARTS = e1.N_RESTARTS_NULL          # = 3, matched to the E1 null for an unbiased p-value
M = 999
ALPHA = 0.05
N_WINDOWS = 12
BONFERRONI = ALPHA / N_WINDOWS           # 0.004167
THIN_WINDOWS = {"2022-Q3", "2022-Q4"}    # reduced-confidence for INTERPRETATION (thin occupancy)
SCAN_SEED = 20260715
FAMILY_JACCARD = 0.30                    # min consecutive-window overlap to call a recurring family
COLORS = {"score": "#1f77b4", "q_hat": "#d62728"}


# ---------------------------------------------------------------------------
# subgroup set helpers (trivial set ops — not method logic, so defined locally)
# ---------------------------------------------------------------------------

def flatten(S):
    return {(f, str(v)) for f, vals in S.items() for v in vals}


def jaccard(A, B):
    a, b = flatten(A), flatten(B)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def subset_stats(window_df, subset):
    sub = mdss.cells_in_subset(window_df, FEATURES, subset)
    C = float(sub["observed"].sum())
    B = float(sub["expected"].sum())
    return C, B, (C / B if B > 0 else np.nan), len(sub)


def fmt_subset(S):
    return "; ".join(f"{f}={vals}" for f, vals in sorted(S.items())) if S else "(none)"


# ===========================================================================
# PART 1 — per-window discovery scan
# ===========================================================================

def part1_discovery(anchored, windows, null_by_window):
    print("\n" + "=" * 78)
    print(f"PART 1 — PER-WINDOW DISCOVERY SCAN (real data, n_restarts={N_RESTARTS} matched to E1 null)")
    print("=" * 78)
    rows, subsets = [], {}
    for w_idx, w in enumerate(windows):
        wdf = anchored[anchored["window_label"] == w][FEATURES + ["observed", "expected"]].reset_index(drop=True)
        subset, score = mdss.scan(wdf, "observed", "expected", FEATURES, n_restarts=N_RESTARTS, seed=SCAN_SEED + w_idx)
        p = e1.pvalue(score, null_by_window[w])
        C, B, q, ncells = subset_stats(wdf, subset)
        subsets[w] = subset
        rows.append({
            "window": w, "subset": repr(subset), "n_features_restricted": len(subset),
            "score": score, "C_observed": C, "B_expected": B, "q_hat": q,
            "n_cells": ncells, "claims_covered": int(C), "window_total": int(wdf["observed"].sum()),
            "pvalue": p, "sig_raw": bool(p < ALPHA), "sig_bonferroni": bool(p < BONFERRONI),
            "thin_occupancy": w in THIN_WINDOWS,
        })
    findings = pd.DataFrame(rows)
    findings.to_csv(config.REPORTS_DIR / "e3_discovery_findings.csv", index=False)

    print(f"\n{'window':9s} {'score':>8s} {'q_hat':>6s} {'C':>6s} {'B':>7s} {'cells':>5s} {'p':>7s}  raw  bonf  thin  subgroup")
    for _, x in findings.iterrows():
        print(f"{x['window']:9s} {x['score']:8.2f} {x['q_hat']:6.2f} {x['C_observed']:6.0f} {x['B_expected']:7.1f} "
              f"{x['n_cells']:5d} {x['pvalue']:7.4f}  {'Y' if x['sig_raw'] else '.':>3s}  {'Y' if x['sig_bonferroni'] else '.':>4s}  "
              f"{'!' if x['thin_occupancy'] else '.':>4s}  {fmt_subset(ast.literal_eval(x['subset']))[:70]}")
    n_raw = int(findings["sig_raw"].sum())
    n_bonf = int(findings["sig_bonferroni"].sum())
    print(f"\nSignificant at raw alpha=0.05: {n_raw}/12 windows.  Survive Bonferroni (p<{BONFERRONI:.4f}): {n_bonf}/12.")
    return findings, subsets


# ===========================================================================
# PART 2 — persistence across consecutive windows
# ===========================================================================

def part2_persistence(anchored, windows, subsets, findings):
    print("\n" + "=" * 78)
    print("PART 2 — PERSISTENCE ACROSS WINDOWS (the emergence signature)")
    print("=" * 78)
    rows = []
    for i, w in enumerate(windows):
        prev = windows[i - 1] if i > 0 else None
        ov = jaccard(subsets[w], subsets[prev]) if prev else np.nan
        f = findings[findings["window"] == w].iloc[0]
        rows.append({"window": w, "subset": fmt_subset(subsets[w]), "pvalue": f["pvalue"],
                     "sig_raw": f["sig_raw"], "overlap_prev": ov})
    persistence = pd.DataFrame(rows)
    persistence.to_csv(config.REPORTS_DIR / "e3_persistence.csv", index=False)
    print(f"\n{'window':9s} {'p':>7s} {'overlap_prev':>12s}  subgroup")
    for _, x in persistence.iterrows():
        ov = f"{x['overlap_prev']:.2f}" if pd.notna(x["overlap_prev"]) else "  — "
        print(f"{x['window']:9s} {x['pvalue']:7.4f} {ov:>12s}  {x['subset'][:66]}")

    # maximal runs of consecutive windows with overlap > FAMILY_JACCARD
    families = []
    run = [windows[0]]
    for i in range(1, len(windows)):
        if jaccard(subsets[windows[i]], subsets[windows[i - 1]]) > FAMILY_JACCARD:
            run.append(windows[i])
        else:
            if len(run) >= 2:
                families.append(list(run))
            run = [windows[i]]
    if len(run) >= 2:
        families.append(list(run))

    print(f"\nRecurring families (>=2 consecutive windows with Jaccard > {FAMILY_JACCARD}): {len(families)}")
    for fam in families:
        sig = any(findings[findings.window == w]["sig_raw"].iloc[0] for w in fam)
        print(f"  {fam[0]}..{fam[-1]} ({len(fam)} windows){' [contains a significant window]' if sig else ''}: "
              f"{fmt_subset(subsets[fam[0]])[:60]}")

    plot_trajectories(anchored, windows, subsets, findings, families)
    return persistence, families


def plot_trajectories(anchored, windows, subsets, findings, families, out_path=None):
    """For each persistent family, hold its subgroup FIXED (the definition from the
    family's most-significant window) and trace that subgroup's score and q_hat across
    ALL 12 windows. A rising trajectory of a FIXED subgroup is the emergence signature
    (unlike the per-window rank-1, which changes subgroup each window and is dominated
    by one-off near-zero-expected spikes)."""
    xs = list(range(len(windows)))
    fam_colors = ["#1f77b4", "#5B8C5A", "#8862AE", "#d62728"]
    fig, (axS, axQ) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if not families:
        families = []
    for fi, fam in enumerate(families):
        fam_f = findings[findings["window"].isin(fam)].sort_values("pvalue")
        rep_w = fam_f.iloc[0]["window"]
        S = subsets[rep_w]
        scores, qs = [], []
        for w in windows:
            wdf = anchored[anchored["window_label"] == w][FEATURES + ["observed", "expected"]].reset_index(drop=True)
            C, B, q, _ = subset_stats(wdf, S)
            scores.append(mdss.score(C, B))
            qs.append(q if pd.notna(q) else 0.0)
        color = fam_colors[fi % len(fam_colors)]
        label = f"{fam[0]}..{fam[-1]} (def@{rep_w}): {fmt_subset(S)[:42]}"
        axS.plot(xs, scores, marker="o", color=color, label=label)
        axQ.plot(xs, qs, marker="s", color=color, label=label)
        i0, i1 = windows.index(fam[0]), windows.index(fam[-1])
        for ax in (axS, axQ):
            ax.axvspan(i0 - 0.3, i1 + 0.3, color=color, alpha=0.06, zorder=0)

    for ax in (axS, axQ):
        ax.set_xticks(xs)
        ax.spines[["top", "right"]].set_visible(False)
    axQ.set_xticklabels(windows, rotation=30, ha="right")
    fig.canvas.draw()  # realize tick labels before recolouring (shared x-axis)
    for w in THIN_WINDOWS:
        if w in windows:
            axQ.get_xticklabels()[windows.index(w)].set_color("#d62728")
    axS.set_ylabel("fixed-subgroup score  F(C,B)")
    axQ.set_ylabel("fixed-subgroup q_hat = C/B")
    axS.set_title("E3 persistent-family emergence: a FIXED subgroup's score & q_hat across the test period "
                  "(shaded = family span; red x-labels = thin windows)")
    axS.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    out = out_path if out_path is not None else config.REPORTS_DIR / "e3_trajectory.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out}")


# ===========================================================================
# PART 3 — caveats: thin-occupancy tags + core-vs-periphery view
# ===========================================================================

def part3_caveats(anchored, windows, subsets, findings):
    print("\n" + "=" * 78)
    print("PART 3 — CAVEATS APPLIED TO FINDINGS")
    print("=" * 78)
    print("THIN-OCCUPANCY: 2022-Q3/Q4 are reduced-confidence for interpretation (a single cell can move\n"
          "the scan); any finding appearing ONLY in those windows is tagged thin-occupancy-caution.")
    print("OVER-SELECTION: E2 showed this scan family has HIGH recall (0.81-0.98) but LOW precision\n"
          "(0.15-0.26) — a flagged subgroup CONTAINS the driver but over-includes noise. So read each\n"
          "subgroup as an UPPER BOUND on the driver's extent; the 'core' cells below (ranked by own\n"
          "observed/expected) are the high-signal part, the rest is swept-in periphery.")

    core_rows = []
    sig = findings[findings["sig_raw"]]
    print(f"\nCore-vs-periphery for the {len(sig)} raw-significant window findings:")
    for _, x in sig.iterrows():
        w = x["window"]
        S = ast.literal_eval(x["subset"])
        wdf = anchored[anchored["window_label"] == w][FEATURES + ["observed", "expected"]].reset_index(drop=True)
        sub = mdss.cells_in_subset(wdf, FEATURES, S).copy()
        sub = sub[sub["expected"] > 0]
        sub["ratio"] = sub["observed"] / sub["expected"]
        sub = sub.sort_values("ratio", ascending=False)
        thin_tag = " [THIN-OCCUPANCY CAUTION]" if w in THIN_WINDOWS else ""
        print(f"\n  {w} (q_hat={x['q_hat']:.2f}, {x['n_cells']} cells, p={x['pvalue']:.4f}){thin_tag}")
        print(f"    core cells (top by observed/expected):")
        for _, c in sub.head(5).iterrows():
            desc = ", ".join(f"{f}={c[f]}" for f in ["peril_type", "risk_code", "loss_census_division"])
            print(f"      obs={c['observed']:.0f} exp={c['expected']:.2f} ratio={c['ratio']:.1f}  ({desc})")
        core_share = sub.head(5)["observed"].sum() / sub["observed"].sum() if len(sub) else np.nan
        print(f"    -> top-5 cells hold {core_share:.0%} of the subgroup's claims (rest is periphery)")
        for _, c in sub.head(5).iterrows():
            core_rows.append({"window": w, "obs": c["observed"], "exp": c["expected"], "ratio": c["ratio"],
                              **{f: c[f] for f in FEATURES}})
    if core_rows:
        pd.DataFrame(core_rows).to_csv(config.REPORTS_DIR / "e3_core_cells.csv", index=False)
    return sig


# ===========================================================================
# PART 4 — new-level cells (complementary finding class)
# ===========================================================================

def part4_new_levels(new_level, anchored, windows):
    print("\n" + "=" * 78)
    print("PART 4 — NEW-LEVEL CELLS (complementary; unreachable by the scan BY CONSTRUCTION)")
    print("=" * 78)
    print("These cells use feature levels ABSENT from training (e.g. OPEN MARKET - FACULTATIVE\n"
          "REINSURANCE) so they have no fitted expected count and are structurally unscoreable by the\n"
          "expectation-based scan — a stated METHOD BOUNDARY, not an oversight. They are themselves\n"
          "candidate emerging drivers, reported qualitatively by test-period volume + trajectory.")

    # window-by-window counts from the REAL test data (read-only)
    raw = diag.load_data(config.ACTIVE_DATA_PATH)
    _m, widx = diag.build_time_indices(raw)
    raw = raw.copy()
    raw["window_idx"] = widx
    w2label = anchored[["window_idx", "window_label"]].drop_duplicates().set_index("window_idx")["window_label"].to_dict()
    raw["window_label"] = raw["window_idx"].map(w2label)
    raw = raw[raw["window_label"].notna()]
    for c in FEATURES:
        raw[c] = raw[c].astype(str)
    counts = raw.groupby(FEATURES + ["window_label"]).size().rename("n").reset_index()

    nl = new_level.copy()
    for c in FEATURES:
        nl[c] = nl[c].astype(str)
    top = nl.sort_values("total_test_observed", ascending=False).head(10).reset_index(drop=True)

    out_rows = []
    print(f"\nTop {len(top)} new-level cells by test-period volume, with window-by-window counts:")
    for _, cell in top.iterrows():
        key = {f: cell[f] for f in FEATURES}
        sub = counts
        for f in FEATURES:
            sub = sub[sub[f] == key[f]]
        per_w = {w: int(sub[sub.window_label == w]["n"].sum()) for w in windows}
        traj = " ".join(f"{per_w[w]:>2d}" for w in windows)
        first_w = next((w for w in windows if per_w[w] > 0), "—")
        last_w = next((w for w in reversed(windows) if per_w[w] > 0), "—")
        desc = f"{cell['peril_type']} x {cell['placing_basis_group']} x {cell['loss_census_division']}"
        print(f"  total={cell['total_test_observed']:>3d} [{first_w}->{last_w}]  {desc[:58]}")
        print(f"      per-window: {traj}")
        out_rows.append({**{f: cell[f] for f in FEATURES}, "total_test_observed": cell["total_test_observed"],
                         "unseen_features": cell["unseen_features"], "first_window": first_w, "last_window": last_w,
                         **{f"n_{w}": per_w[w] for w in windows}})
    pd.DataFrame(out_rows).to_csv(config.REPORTS_DIR / "e3_new_level_findings.csv", index=False)
    print(f"\n(window order: {' '.join(w[2:] for w in windows)})")
    return top


# ===========================================================================
# PART 5 — cross-method concordance (secondary; FGSS only — see note)
# ===========================================================================

def part5_concordance(anchored, windows, subsets, findings):
    print("\n" + "=" * 78)
    print("PART 5 — CROSS-METHOD CONCORDANCE (secondary)")
    print("=" * 78)
    fgss_null_path = config.REPORTS_DIR / "e2_fgss_null_maxscores.csv"
    if not fgss_null_path.exists():
        print("  FGSS null not found; skipping concordance.")
        return None
    print("Only FGSS is run here: its detect reads the SAME anchored expected the proposed method uses,\n"
          "so it is a genuine one-call check. MA and EPM are OMITTED — their detect requires constructing\n"
          "a trailing real-data REFERENCE PERIOD (pooled prior-window counts), which is more than calling\n"
          "an existing detect function; per the task, concordance is skipped where it needs more than that.")
    mod11 = _load_module("mod11", _scripts_dir / "11_fgss.py")
    alpha = mod11.mod08.alpha_from_e1()
    method = mod11.make_fgss_method(alpha)
    fnull = pd.read_csv(fgss_null_path)
    fgss_null = {w: g["max_bj_score"].to_numpy() for w, g in fnull.groupby("window_label")}

    rows = []
    for w_idx, w in enumerate(windows):
        wdf = anchored[anchored["window_label"] == w][FEATURES + ["observed", "expected"]].reset_index(drop=True)
        exp = mod11.fgss_fit_expected(wdf.assign(share=wdf["expected"] / wdf["expected"].sum()))
        det, sub, sc, p = method.detect(wdf, exp, fgss_null[w], N_RESTARTS, SCAN_SEED + w_idx)
        ov = jaccard(sub, subsets[w])
        rows.append({"window": w, "fgss_pvalue": p, "fgss_sig": bool(det),
                     "proposed_sig": bool(findings[findings.window == w]["sig_raw"].iloc[0]),
                     "subgroup_overlap": ov})
    conc = pd.DataFrame(rows)
    conc.to_csv(config.REPORTS_DIR / "e3_concordance.csv", index=False)
    print(f"\n{'window':9s} {'proposed_sig':>12s} {'fgss_sig':>9s} {'fgss_p':>7s} {'overlap':>8s}")
    for _, x in conc.iterrows():
        print(f"{x['window']:9s} {'Y' if x['proposed_sig'] else '.':>12s} {'Y' if x['fgss_sig'] else '.':>9s} "
              f"{x['fgss_pvalue']:7.4f} {x['subgroup_overlap']:8.2f}")
    both = conc[(conc.proposed_sig) & (conc.fgss_sig)]
    print(f"\nWindows flagged by BOTH proposed + FGSS: {len(both)} "
          f"({', '.join(both['window'].tolist()) if len(both) else 'none'}); "
          "concordant flags are a robustness signal; disagreement is informative (methods have different power profiles).")
    return conc


# ===========================================================================
# PART 6 — exposure-check handoff table
# ===========================================================================

def part6_handoff(anchored, windows, subsets, findings):
    print("\n" + "=" * 78)
    print("PART 6 — EXPOSURE-CHECK HANDOFF TABLE (for the industry supervisor)")
    print("=" * 78)
    print("PRIMARY LIMITATION: real-data findings cannot separate genuine RISK growth from EXPOSURE /\n"
          "portfolio growth. This table hands each significant subgroup + its full trajectory to the\n"
          "supervisor to validate against exposure data.")
    sig = findings[findings["sig_raw"]]
    rows = []
    for _, x in sig.iterrows():
        wsig = x["window"]
        S = subsets[wsig]
        # trajectory of THIS subgroup across ALL windows (rising q_hat => emergence)
        for w in windows:
            wdf = anchored[anchored["window_label"] == w][FEATURES + ["observed", "expected"]].reset_index(drop=True)
            C, B, q, ncells = subset_stats(wdf, S)
            rows.append({"finding_window": wsig, "subgroup": fmt_subset(S),
                         "trajectory_window": w, "observed_claims": int(C), "expected_claims": round(B, 2),
                         "q_hat": round(q, 3) if pd.notna(q) else np.nan,
                         "is_finding_window": (w == wsig), "thin_occupancy_window": w in THIN_WINDOWS})
    handoff = pd.DataFrame(rows)
    handoff.to_csv(config.REPORTS_DIR / "e3_exposure_check_handoff.csv", index=False)
    print(f"\nHandoff table: {len(sig)} significant subgroup(s) x 12-window trajectory each "
          f"= {len(handoff)} rows -> reports/e3_exposure_check_handoff.csv")
    for _, x in sig.iterrows():
        S = subsets[x["window"]]
        traj = [subset_stats(anchored[anchored.window_label == w][FEATURES + ["observed", "expected"]].reset_index(drop=True), S)[2] for w in windows]
        traj_str = " ".join(f"{q:4.1f}" if pd.notna(q) else "  . " for q in traj)
        print(f"  finding@{x['window']}: {fmt_subset(S)[:55]}")
        print(f"      q_hat trajectory: {traj_str}")
    print(f"  (window order: {' '.join(w[2:] for w in windows)})")
    return handoff


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    artifacts = {
        "processed_data.csv": config.PROCESSED_DATA_PATH,
        config.ACTIVE_DATA_PATH.name: config.ACTIVE_DATA_PATH,
        config.ANCHORED_EXPECTED_COUNTS_PATH.name: config.ANCHORED_EXPECTED_COUNTS_PATH,
        "e1_null_maxscores.csv": config.REPORTS_DIR / "e1_null_maxscores.csv",
        "new_level_cells.csv": config.REPORTS_DIR / "new_level_cells.csv",
    }
    md5_before = {n: mod04.file_md5(p) for n, p in artifacts.items()}

    anchored = pd.read_csv(config.ANCHORED_EXPECTED_COUNTS_PATH)
    windows = sorted(anchored["window_label"].unique(),
                     key=lambda w: anchored.loc[anchored["window_label"] == w, "window_idx"].iloc[0])
    nulldf = pd.read_csv(config.REPORTS_DIR / "e1_null_maxscores.csv")
    null_by_window = {w: g["max_score"].to_numpy() for w, g in nulldf.groupby("window_label")}
    new_level = pd.read_csv(config.REPORTS_DIR / "new_level_cells.csv")
    print(f"E3 DISCOVERY — real test data only (NO injection/grid/synthetic). {len(windows)} windows, "
          f"613 cells/window, M={M} null, n_restarts={N_RESTARTS}.")

    findings, subsets = part1_discovery(anchored, windows, null_by_window)
    persistence, families = part2_persistence(anchored, windows, subsets, findings)
    sig = part3_caveats(anchored, windows, subsets, findings)
    part4_new_levels(new_level, anchored, windows)
    conc = part5_concordance(anchored, windows, subsets, findings)
    part6_handoff(anchored, windows, subsets, findings)

    # ---- summary ----
    print("\n" + "=" * 78)
    print("SUMMARY — E3 DISCOVERY")
    print("=" * 78)
    n_raw, n_bonf = int(findings.sig_raw.sum()), int(findings.sig_bonferroni.sum())
    print(f"\n1. Per-window findings: {n_raw}/12 significant at raw 0.05; {n_bonf}/12 survive Bonferroni "
          f"(p<{BONFERRONI:.4f}).")
    for _, x in findings[findings.sig_raw].iterrows():
        tags = []
        if x["sig_bonferroni"]:
            tags.append("BONFERRONI-ROBUST")
        if x["thin_occupancy"]:
            tags.append("THIN-OCCUPANCY-CAUTION")
        print(f"     {x['window']}: p={x['pvalue']:.4f} q_hat={x['q_hat']:.2f} {'['+', '.join(tags)+']' if tags else ''}"
              f"  {fmt_subset(subsets[x['window']])[:50]}")
    print(f"2. Persistence: {len(families)} recurring family/families (>=2 consecutive windows, Jaccard>{FAMILY_JACCARD}); "
          "trajectories in e3_trajectory.png. A rising q_hat within a family is the emergence signature.")
    thin_only = [x["window"] for _, x in findings[findings.sig_raw].iterrows() if x["thin_occupancy"]]
    print(f"3. Caveats: thin-occupancy findings = {thin_only or 'none'}; every significant subgroup reported as an "
          "UPPER BOUND with a core-vs-periphery breakdown (e3_core_cells.csv) given the scan's low precision.")
    print("4. New-level cells: complementary candidate drivers the scan cannot reach by construction; top by "
          "volume in e3_new_level_findings.csv (largest: Severe Storm-Tornado x OPEN MARKET-FAC REINSURANCE x ESC, 39).")
    if conc is not None:
        both = int(((conc.proposed_sig) & (conc.fgss_sig)).sum())
        print(f"5. Concordance (FGSS only): {both} window(s) flagged by both; MA/EPM omitted (need trailing-reference build).")
    print("6. Exposure handoff: significant subgroups + full 12-window trajectories -> e3_exposure_check_handoff.csv "
          "(the artifact for the supervisor's exposure validation).")

    md5_after = {n: mod04.file_md5(p) for n, p in artifacts.items()}
    all_ok = all(md5_before[n] == md5_after[n] for n in artifacts)
    print("\nNO synthetic/injected data was used anywhere in E3 — real observed vs anchored expected only.")
    print("Source & input artifacts MD5 unchanged:")
    for n in artifacts:
        print(f"  {n}: {md5_before[n] == md5_after[n]}")
    print(f"ALL artifacts unchanged: {all_ok}")


if __name__ == "__main__":
    main()
