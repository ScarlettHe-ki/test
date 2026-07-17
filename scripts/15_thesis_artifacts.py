"""Thesis artifacts — publication-quality vector plots + a cell-level findings table.

ARTIFACT-ONLY: runs NO experiment, NO scan, NO recalibration. It reconstructs each
kept plot's INPUT from the already-saved result CSVs (or, for the two data-chapter
diagnostics, by re-running script 01's read-only aggregation on the frozen data) and
calls the SAME plotting function from its source script with a .pdf output path, so the
figure is a true VECTOR PDF rather than a rasterised PNG->PDF. It also builds a
cell-level discovery-findings table for the supervisor's exposure check.

Read-only on data/ and every result CSV (MD5-verified unchanged before/after). The only
writes are plots/*.pdf and the new reports/e3_findings_by_cell.csv.

MINIMAL PARAMETERIZATION (noted per the task): three plot functions hard-coded their
output path, so an optional `out_path=None` argument was added to each (default =
original behaviour, plot content unchanged): 13.plot_power_with_ci, 13.plot_survival,
14.plot_trajectories. Every other plotting function already takes a `filename` and
builds `config.REPORTS_DIR / filename`; passing an ABSOLUTE .pdf path there redirects
the output cleanly (pathlib: an absolute right-hand operand wins), with no edit needed.
"""

import ast
import contextlib
import importlib.util
import io
import sys
from collections import defaultdict
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


_sd = Path(__file__).resolve().parent
diag = _load_module("diag01", _sd / "01_sparsity_dispersion_check.py")
mod04 = _load_module("mod04", _sd / "04_expected_counts_fit.py")
mod05 = _load_module("mod05", _sd / "05_expected_counts_anchored.py")
mod07 = _load_module("e1_07", _sd / "07_e1_calibration.py")
mod09 = _load_module("mod09", _sd / "09_moving_average_mdss.py")
mod13 = _load_module("mod13", _sd / "13_results_analysis.py")
mod14 = _load_module("mod14", _sd / "14_e3_discovery.py")

FEATURES = mod04.FEATURES
REPORTS = config.REPORTS_DIR
PLOTS = config.ROOT_DIR / "plots"

# calibration-check FPR per window is a diagnostic from the M=999 driver run (07/12);
# it is not stored in the null-scores CSV, so it is carried here to regenerate the exact
# same bar chart. Cross-checkable against the plotted 5% line and the M=999 run log.
CALIB_FPR_M999 = {'2022-Q1':0.043,'2022-Q2':0.061,'2022-Q3':0.040,'2022-Q4':0.053,
                  '2023-Q1':0.054,'2023-Q2':0.054,'2023-Q3':0.051,'2023-Q4':0.057,
                  '2024-Q1':0.047,'2024-Q2':0.067,'2024-Q3':0.051,'2024-Q4':0.055}

THIN_WINDOWS = {"2022-Q3", "2022-Q4"}
NEAR_ZERO_MEAN_EXP = 0.10  # mean expected-per-flagged-window below this = near modelling
# floor (the anchored backoff floor is ~0.0024 claims/cell) -> the "exp=0.00 / ratio~400"
# artifact from the core-vs-periphery view; substantive cells sit well above it.


# ===========================================================================
# TASK 1 — regenerate kept plots as vector PDFs
# ===========================================================================

def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def regenerate_plots():
    PLOTS.mkdir(parents=True, exist_ok=True)
    manifest = []

    def do(pdf_name, chapter, source, func, thunk):
        try:
            thunk()
            manifest.append({"pdf": f"plots/{pdf_name}", "chapter": chapter,
                             "source": source, "function": func, "status": "ok"})
        except Exception as e:
            manifest.append({"pdf": f"plots/{pdf_name}", "chapter": chapter,
                             "source": source, "function": func, "status": f"FAILED: {e}"})

    def P(name):
        return str(PLOTS / name)

    # ---- Data chapter (script 01, re-run read-only aggregation on frozen data) ----
    df = diag.load_data(config.ACTIVE_DATA_PATH)

    def _monthly():
        mt = df.groupby(df["claim_date"].dt.to_period("M")).size().sort_index()
        z = (mt - mt.mean()) / mt.std()
        diag.plot_monthly_totals(mt, z, filename=P("monthly_total_claims.pdf"))
    do("monthly_total_claims.pdf", "Data", "01_sparsity_dispersion_check.py",
       "plot_monthly_totals", _monthly)

    def _cells():
        n_months = _quiet(diag.report_date_column, df)
        scan_cols, cards, geo_feature, _gf = _quiet(diag.identify_categorical_features, df)
        month_idx, window_idx = diag.build_time_indices(df)
        sparsity, _excl, _pk = _quiet(diag.report_sparsity, df, scan_cols, cards, geo_feature,
                                      month_idx, window_idx, n_months)
        diag.plot_cell_distributions(sparsity, filename=P("cell_count_distribution.pdf"))
    do("cell_count_distribution.pdf", "Data", "01_sparsity_dispersion_check.py",
       "plot_cell_distributions", _cells)

    # ---- Methodology (04 fixed-fit residuals, 05 anchoring checks) ----
    def _resid_fixed():
        scored = pd.read_csv(REPORTS / "expected_counts_test.csv")  # has window_idx, pearson_resid
        mod04.plot_residuals(scored, P("residuals_fixed_fit.pdf"),
                             "Mean Pearson residual by test window, fixed fit")
    do("residuals_fixed_fit.pdf", "Methodology", "04_expected_counts_fit.py",
       "plot_residuals", _resid_fixed)

    anchored = pd.read_csv(config.ANCHORED_EXPECTED_COUNTS_PATH)

    def _obs_exp():
        scored = anchored.rename(columns={"observed": "count"})  # plot expects 'count','expected','window_idx'
        mod05.plot_aggregate_check(scored, P("observed_vs_expected_anchored.pdf"))
    do("observed_vs_expected_anchored.pdf", "Methodology", "05_expected_counts_anchored.py",
       "plot_aggregate_check", _obs_exp)

    def _share_resid():
        mod05.plot_share_residual_distributions(anchored, P("share_residuals_early_vs_late.pdf"))
    do("share_residuals_early_vs_late.pdf", "Methodology", "05_expected_counts_anchored.py",
       "plot_share_residual_distributions", _share_resid)

    # ---- E1 ----
    nulldf = pd.read_csv(REPORTS / "e1_null_maxscores.csv")
    e1_windows = sorted(nulldf["window_label"].unique(), key=lambda w: (int(w[:4]), w[-1]))

    def _null_dist():
        null_results = {w: {"null_scores": nulldf.loc[nulldf.window_label == w, "max_score"].to_numpy()}
                        for w in e1_windows}
        mod07.plot_null_distributions(null_results, P("e1_null_distributions.pdf"))
    do("e1_null_distributions.pdf", "E1", "07_e1_calibration.py", "plot_null_distributions", _null_dist)

    def _calib():
        check = {w: CALIB_FPR_M999[w] for w in e1_windows}
        mod07.plot_calibration_check(check, P("e1_calibration_check.pdf"))
    do("e1_calibration_check.pdf", "E1", "07_e1_calibration.py", "plot_calibration_check", _calib)

    # ---- E2 ----
    def _power_ci():
        power_ci = pd.read_csv(REPORTS / "e2_power_with_ci.csv")
        mod13.plot_power_with_ci(power_ci, out_path=P("e2_power_with_ci.pdf"))
    do("e2_power_with_ci.pdf", "E2", "13_results_analysis.py", "plot_power_with_ci", _power_ci)

    def _prj():
        ss = pd.read_csv(REPORTS / "e2_step_summary.csv")
        mod09.plot_precision_recall_jaccard(ss, P("e2_precision_recall_jaccard_by_method.pdf"))
    do("e2_precision_recall_jaccard_by_method.pdf", "E2", "09_moving_average_mdss.py",
       "plot_precision_recall_jaccard", _prj)

    def _survival():
        surv = pd.read_csv(REPORTS / "e2_ramp_survival.csv")
        mod13.plot_survival(surv, int(surv["k_windows"].max()), out_path=P("e2_ramp_survival.pdf"))
    do("e2_ramp_survival.pdf", "E2", "13_results_analysis.py", "plot_survival", _survival)

    # ---- E3 ----
    def _traj():
        findings = pd.read_csv(REPORTS / "e3_discovery_findings.csv")
        windows = sorted(findings["window"].unique(), key=lambda w: (int(w[:4]), w[-1]))
        subsets = {r["window"]: ast.literal_eval(r["subset"]) for _, r in findings.iterrows()}
        families, run = [], [windows[0]]
        for i in range(1, len(windows)):
            if mod14.jaccard(subsets[windows[i]], subsets[windows[i - 1]]) > mod14.FAMILY_JACCARD:
                run.append(windows[i])
            else:
                if len(run) >= 2:
                    families.append(list(run))
                run = [windows[i]]
        if len(run) >= 2:
            families.append(list(run))
        mod14.plot_trajectories(anchored, windows, subsets, findings, families, out_path=P("e3_trajectory.pdf"))
    do("e3_trajectory.pdf", "E3", "14_e3_discovery.py", "plot_trajectories", _traj)

    return manifest


def verify_vector(pdf_path):
    """Heuristic: matplotlib line/bar/hist/scatter PDFs are pure vector paths and embed
    NO raster image XObject; a PNG->PDF conversion would embed one (/Subtype /Image)."""
    data = pdf_path.read_bytes()
    is_pdf = data[:5] == b"%PDF-"
    has_raster = (b"/Subtype /Image" in data) or (b"/Subtype/Image" in data)
    return is_pdf and not has_raster


# ===========================================================================
# TASK 2 — cell-level discovery findings table
# ===========================================================================

def build_cell_table():
    findings = pd.read_csv(REPORTS / "e3_discovery_findings.csv")
    conc = pd.read_csv(REPORTS / "e3_concordance.csv")
    anchored = pd.read_csv(config.ANCHORED_EXPECTED_COUNTS_PATH)

    # "flagged" = raw alpha=0.05 (survives_bonferroni carried as a separate boolean).
    flagged = findings[findings["sig_raw"]]
    bonf_windows = set(findings.loc[findings["sig_bonferroni"], "window"])
    fgss_windows = set(conc.loc[conc["fgss_sig"], "window"])

    # accumulate per unique CELL SIGNATURE (restricted features -> concrete value,
    # unrestricted features -> "ALL"), summing over concrete cells and flagged windows.
    acc = defaultdict(lambda: {"obs": 0.0, "exp": 0.0, "windows": set()})
    for _, row in flagged.iterrows():
        w = row["window"]
        S = ast.literal_eval(row["subset"])
        restricted = set(S.keys())
        wdf = anchored[anchored["window_label"] == w]
        mask = pd.Series(True, index=wdf.index)
        for f, vals in S.items():
            mask &= wdf[f].isin(vals)
        cells = wdf[mask & (wdf["observed"] > 0)]          # skip empty combinations
        for c in cells.to_dict("records"):
            sig = tuple((f, (str(c[f]) if f in restricted else "ALL")) for f in FEATURES)
            a = acc[sig]
            a["obs"] += c["observed"]
            a["exp"] += c["expected"]
            a["windows"].add(w)

    rows = []
    for sig, a in acc.items():
        d = dict(sig)
        wins = sorted(a["windows"], key=lambda w: (int(w[:4]), w[-1]))
        mean_exp = a["exp"] / len(wins) if wins else np.nan
        rows.append({
            **d,
            "total_test_observed": int(round(a["obs"])),
            "expected_flagged_windows": round(a["exp"], 4),
            "q_hat": round(a["obs"] / a["exp"], 2) if a["exp"] > 0 else np.nan,
            "n_windows_flagged": len(wins),
            "first_window": wins[0], "last_window": wins[-1],
            "survives_bonferroni": any(w in bonf_windows for w in wins),
            "thin_occupancy_caution": set(wins).issubset(THIN_WINDOWS),
            "near_zero_expected_artifact": bool(mean_exp < NEAR_ZERO_MEAN_EXP),
            "fgss_corroborated": any(w in fgss_windows for w in wins),
        })
    cols = FEATURES + ["total_test_observed", "expected_flagged_windows", "q_hat",
                       "n_windows_flagged", "first_window", "last_window",
                       "survives_bonferroni", "thin_occupancy_caution",
                       "near_zero_expected_artifact", "fgss_corroborated"]
    tbl = pd.DataFrame(rows)[cols].sort_values("total_test_observed", ascending=False).reset_index(drop=True)
    tbl.to_csv(REPORTS / "e3_findings_by_cell.csv", index=False)
    return tbl


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    md5_targets = {
        "processed_data.csv": config.PROCESSED_DATA_PATH,
        config.ACTIVE_DATA_PATH.name: config.ACTIVE_DATA_PATH,
        "expected_counts_anchored_test.csv": config.ANCHORED_EXPECTED_COUNTS_PATH,
    }
    for n in ["expected_counts_test.csv", "e1_null_maxscores.csv", "e2_step_summary.csv",
              "e2_power_with_ci.csv", "e2_ramp_survival.csv", "e3_discovery_findings.csv",
              "e3_core_cells.csv", "e3_concordance.csv", "e3_exposure_check_handoff.csv",
              "e3_new_level_findings.csv"]:
        md5_targets[n] = REPORTS / n
    md5_before = {n: mod04.file_md5(p) for n, p in md5_targets.items()}

    print("=" * 78)
    print("TASK 1 — REGENERATE KEPT THESIS PLOTS AS VECTOR PDFs")
    print("=" * 78)
    print("\nSuperseded PNG variants DROPPED (earlier design iterations, not the frozen "
          "8-feature / loss_census_division / processed_data_region_clean.csv design):")
    dropped = {
        "monthly_total_claims_geo_kept.png / _region_coarse.png":
            "geo-exploration (03) and region-coarsening (00) variants; frozen plot is script 01's base name on ACTIVE_DATA_PATH",
        "cell_count_distribution_geo_kept.png / _region_coarse.png / _grid_subclass_variant.png":
            "geo (03) / region-coarse (00) / sub_class-grid (02) variants; frozen is 01's base name",
        "observed_vs_expected_fixed_fit.png / residuals_expanding_refit.png / occupancy_floor_test.png":
            "superseded fixed-fit / expanding-refit / occupancy diagnostics not in the keeper list",
        "e2_power_curves(_by_method).png / e2_time_to_detect(_by_method).png / e2_jaccard_recall_curves.png / e2_power_per_window.png":
            "superseded by CI/survival versions (e2_power_with_ci, e2_ramp_survival)",
    }
    for k, v in dropped.items():
        print(f"  DROP  {k}\n         -> {v}")

    manifest = regenerate_plots()
    print("\nPLOT MANIFEST (kept -> vector PDF):")
    print(f"  {'pdf':52s} {'chapter':11s} source::function")
    for m in manifest:
        print(f"  {m['pdf']:52s} {m['chapter']:11s} {m['source']}::{m['function']}  [{m['status']}]")

    print("\nVector (not raster) verification:")
    all_vec = True
    for m in manifest:
        p = config.ROOT_DIR / m["pdf"]
        if not p.exists():
            print(f"  {m['pdf']}: MISSING"); all_vec = False; continue
        vec = verify_vector(p)
        all_vec &= vec
        print(f"  {m['pdf']}: {'VECTOR ok' if vec else 'RASTER?! check'} ({p.stat().st_size//1024} KB)")

    print("\n" + "=" * 78)
    print("TASK 2 — CELL-LEVEL DISCOVERY FINDINGS TABLE")
    print("=" * 78)
    print("Built by EXPANDING each raw-flagged subgroup into its concrete constituent cells against\n"
          "the anchored counts (e3_core_cells.csv holds only the top-5 core cells per window, so it\n"
          "is insufficient for full expansion). Conjunction-across / disjunction-within semantics:\n"
          "one row per concrete cell (cross-product of restricted values); unrestricted features = 'ALL'.")
    tbl = build_cell_table()
    handoff = pd.read_csv(REPORTS / "e3_exposure_check_handoff.csv")
    print(f"\nSaved reports/e3_findings_by_cell.csv: {len(tbl)} rows x {tbl.shape[1]} columns.")
    print(f"Columns: {list(tbl.columns)}")
    print(f"\nRelationship to e3_exposure_check_handoff.csv ({len(handoff)} rows = "
          f"{handoff['finding_window'].nunique()} subgroups x 12-window trajectories): the handoff is\n"
          "SUBGROUP-level (one subgroup's trajectory per row); this new table is CELL-level (one\n"
          "concrete cell per row) — they complement each other, neither replaces the other.")
    print("\nFlag counts:")
    print(f"  thin_occupancy_caution      : {int(tbl['thin_occupancy_caution'].sum())} / {len(tbl)}")
    print(f"  near_zero_expected_artifact : {int(tbl['near_zero_expected_artifact'].sum())} / {len(tbl)}  "
          f"(mean expected-per-flagged-window < {NEAR_ZERO_MEAN_EXP})")
    print(f"  fgss_corroborated           : {int(tbl['fgss_corroborated'].sum())} / {len(tbl)}")
    print(f"  survives_bonferroni         : {int(tbl['survives_bonferroni'].sum())} / {len(tbl)}")
    print("\nTop 10 cells by total_test_observed:")
    show = FEATURES[:1] + ["risk_code", "loss_census_division", "placing_basis_group",
                           "total_test_observed", "q_hat", "n_windows_flagged",
                           "survives_bonferroni", "near_zero_expected_artifact", "fgss_corroborated"]
    with pd.option_context("display.width", 240, "display.max_colwidth", 34):
        print(tbl[show].head(10).to_string(index=False))

    print("\n" + "=" * 78)
    print("SUMMARY / INTEGRITY")
    print("=" * 78)
    ok = sum(1 for m in manifest if m["status"] == "ok")
    print(f"Plots regenerated as vector PDF: {ok}/{len(manifest)} (all vector: {all_vec}).")
    print("No experiment / scan / recalibration was run — plot inputs were reconstructed from the\n"
          "existing result CSVs (and script 01's read-only aggregation for the two data-chapter plots).")
    md5_after = {n: mod04.file_md5(p) for n, p in md5_targets.items()}
    unchanged = all(md5_before[n] == md5_after[n] for n in md5_targets)
    print(f"\nSource-artifact MD5 unchanged (data + all result CSVs): {unchanged}")
    for n in md5_targets:
        if md5_before[n] != md5_after[n]:
            print(f"  CHANGED: {n}")
    print(f"New artifacts written: {len(manifest)} PDFs in plots/ + reports/e3_findings_by_cell.csv "
          "(the only new/modified outputs).")


if __name__ == "__main__":
    main()
