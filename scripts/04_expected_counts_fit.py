"""Fit the expected-count baseline that MDSS will scan against: main-effects
Poisson/negative-binomial GLM on (cell x 3-month window) counts, trained on
2018-2020, scored on 2022-2024, with a staleness diagnostic on the residuals
and a per-window occupancy floor check.

Frozen design (from 00-03): 8 scan features (peril_type, syndicate,
group_class, placing_basis_group, leader_status, new_renewal, risk_code,
loss_census_division), 3-month detection window, negative binomial null.

Read-only on config.ACTIVE_DATA_PATH and config.PROCESSED_DATA_PATH (writes
no data files, only reports/). Reuses build_time_indices/cell_counts/
cell_distribution from 01_sparsity_dispersion_check.py.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

import config


def file_md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = _load_module(
    "diag01", Path(__file__).resolve().parent / "01_sparsity_dispersion_check.py"
)

FEATURES = [
    "peril_type", "syndicate", "group_class", "placing_basis_group",
    "leader_status", "new_renewal", "risk_code", "loss_census_division",
]
TRAIN_YEARS = {2018, 2019, 2020}
BUFFER_YEARS = {2021}
TEST_YEARS = {2022, 2023, 2024}

# Recorded from 03_geo_kept_sparsity_check.py's frozen-design sanity check
# (full dataset, 8 features, 3-month window).
FROZEN_DESIGN_REFERENCE = {"n_populated_cells": 3201, "mean_per_cell": 3.280}

FORMULA = "count ~ " + " + ".join(f"C({c})" for c in FEATURES) + " + C(quarter) + window_idx"
OCCUPANCY_WARN_THRESHOLD = diag.MEAN_PER_CELL_USABLE_THRESHOLD  # 3.0
OCCUPANCY_CRITICAL_THRESHOLD = 2.5

REFIT_STAGES = [
    {"predict_year": 2022, "train_years": [2018, 2019, 2020]},
    {"predict_year": 2023, "train_years": [2018, 2019, 2020, 2022]},
    {"predict_year": 2024, "train_years": [2018, 2019, 2020, 2022, 2023]},
]


def window_label(window_idx):
    year = 2018 + window_idx // 4
    quarter = window_idx % 4 + 1
    return f"{year}-Q{quarter}"


def resolve_train_test_split(df):
    print("=" * 78)
    print("STEP 0 — RESOLVE THE 2021 GAP AND THE CELL GRID")
    print("=" * 78)

    print(f"\nMonth range present: {df['claim_date'].min().date()} -> {df['claim_date'].max().date()}")
    years_present = sorted(int(y) for y in df["claim_date"].dt.year.unique())
    print(f"Years present: {years_present}")
    print(f"Rows per year:\n{df['claim_date'].dt.year.value_counts().sort_index().to_string()}")

    print(
        f"\nTrain = {sorted(TRAIN_YEARS)}, buffer = {sorted(BUFFER_YEARS)} (held out), "
        f"test = {sorted(TEST_YEARS)}."
    )
    print(
        "Chosen: 2021 is a BUFFER, excluded from both train and test. "
        "Reason: it keeps the train/test gap unambiguous, which sharpens the "
        "staleness diagnostic this script exists to run (Step 3) — the whole "
        "point is testing whether a fit on the flat 2018-2020 era goes stale "
        "by the time it reaches 2022-2024, and a buffer year removes any "
        "argument that the model 'saw' something close to the test period."
    )

    train_mask = df["claim_date"].dt.year.isin(TRAIN_YEARS)
    test_mask = df["claim_date"].dt.year.isin(TEST_YEARS)
    buffer_mask = df["claim_date"].dt.year.isin(BUFFER_YEARS)
    print(
        f"\nRow counts: train={train_mask.sum()}, buffer={buffer_mask.sum()}, "
        f"test={test_mask.sum()} (total {len(df)})"
    )
    return train_mask, test_mask, buffer_mask


def sanity_check_cell_grid(df, month_idx, window_idx):
    counts = diag.cell_counts(df, FEATURES, window_idx)
    dist = diag.cell_distribution(counts)
    print(
        f"\nFull-dataset cell grid (8 features x 3-month window): "
        f"{dist['n_populated_cells']} populated cells, mean/cell = {dist['mean_per_cell']:.3f}"
    )
    ref = FROZEN_DESIGN_REFERENCE
    matches = (
        dist["n_populated_cells"] == ref["n_populated_cells"]
        and abs(dist["mean_per_cell"] - ref["mean_per_cell"]) < 0.005
    )
    print(
        f"Matches frozen-design reference ({ref['n_populated_cells']} cells, "
        f"{ref['mean_per_cell']:.3f} mean/cell): {'PASS' if matches else 'MISMATCH — investigate'}"
    )
    if not matches:
        raise AssertionError("Cell grid does not match the frozen design; feature set may be wrong.")


def build_scaffold(sub_df, cells, windows):
    scaffold = cells.merge(pd.DataFrame({"window_idx": windows}), how="cross")
    scaffold["quarter"] = (scaffold["window_idx"] % 4).astype(int)
    obs = sub_df.groupby(FEATURES + ["window_idx"]).size().reset_index(name="count")
    scaffold = scaffold.merge(obs, on=FEATURES + ["window_idx"], how="left")
    scaffold["count"] = scaffold["count"].fillna(0).astype(int)
    return scaffold


def fit_expected_count_model(scaffold, formula=FORMULA, verbose=True):
    pois = smf.glm(formula, data=scaffold, family=sm.families.Poisson()).fit()

    mu = pois.fittedvalues
    y = scaffold["count"]
    ct_y = ((y - mu) ** 2 - y) / mu
    ct_fit = sm.OLS(ct_y, mu).fit(cov_type="HC1")
    alpha_hat = max(ct_fit.params.iloc[0], 1e-6)
    pearson_ratio = pois.pearson_chi2 / pois.df_resid

    nb = sm.GLM(y, np.asarray(pois.model.exog), family=sm.families.NegativeBinomial(alpha=alpha_hat)).fit()

    if verbose:
        print(f"\nTraining rows (scaffold): {len(scaffold)}, unique cells: {len(scaffold[FEATURES].drop_duplicates())}")
        print(f"Poisson Pearson chi2/df = {pearson_ratio:.3f} (>>1 indicates overdispersion)")
        print(
            f"Cameron-Trivedi overdispersion test: alpha_hat = {alpha_hat:.4f}, "
            f"t = {ct_fit.tvalues.iloc[0]:.3f}, p = {ct_fit.pvalues.iloc[0]:.2e} "
            f"({'REJECT equidispersion -> NB justified' if ct_fit.pvalues.iloc[0] < 0.05 else 'fails to reject equidispersion'})"
        )
        print("\nTrend and seasonality coefficients (Poisson fit):")
        for name in pois.params.index:
            if name == "window_idx" or "quarter" in name:
                print(f"  {name:20s} coef={pois.params[name]:+.4f}  se={pois.bse[name]:.4f}  p={pois.pvalues[name]:.4f}")
        if "window_idx" in pois.params.index:
            print(f"  -> trend: {pois.params['window_idx']:+.4f} per 3-month window "
                  f"({pois.params['window_idx']*4:+.4f} annualized, log scale)")

    return {"pois": pois, "nb": nb, "alpha": alpha_hat, "pearson_ratio": pearson_ratio, "ct_pvalue": ct_fit.pvalues.iloc[0]}


def score_cells(bundle, train_levels, target_df, target_windows, verbose=True):
    target_cells = target_df[FEATURES].drop_duplicates()
    scoreable = target_cells.apply(lambda r: all(r[c] in train_levels[c] for c in FEATURES), axis=1)
    n_dropped = (~scoreable).sum()
    if verbose and n_dropped:
        dropped_rows = target_df.merge(target_cells[~scoreable], on=FEATURES, how="inner")
        print(
            f"\n{n_dropped}/{len(target_cells)} target cells contain a feature level unseen "
            f"in training and cannot be scored by the frozen model "
            f"({len(dropped_rows)} rows / {len(target_df)} excluded, "
            f"{len(dropped_rows) / len(target_df):.1%})."
        )

    scaffold = build_scaffold(target_df, target_cells[scoreable], target_windows)
    X = patsy.dmatrix(bundle["pois"].model.data.design_info, scaffold, return_type="dataframe")
    scaffold["expected_pois"] = bundle["pois"].predict(scaffold)
    scaffold["expected_nb"] = bundle["nb"].predict(np.asarray(X))
    scaffold["resid"] = scaffold["count"] - scaffold["expected_nb"]
    scaffold["pearson_resid"] = scaffold["resid"] / np.sqrt(
        scaffold["expected_nb"] + scaffold["expected_nb"] ** 2 * bundle["alpha"]
    )
    scaffold["window_label"] = scaffold["window_idx"].map(window_label)
    return scaffold


def plot_observed_vs_expected(scored, filename, title):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    agg = scored.groupby("window_idx").agg(observed=("count", "sum"), expected=("expected_nb", "sum")).sort_index()
    labels = [window_label(w) for w in agg.index]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(labels, agg["observed"], color=diag.COLOR_OUTLIER, linewidth=2, marker="o", label="observed")
    ax.plot(labels, agg["expected"], color=diag.COLOR_LINE, linewidth=2, marker="o", linestyle="--", label="expected (fixed fit)")
    ax.set_title(title)
    ax.set_xlabel("3-month test window")
    ax.set_ylabel("Aggregate claim count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_residuals(scored, filename, title):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    agg = scored.groupby("window_idx")["pearson_resid"].mean().sort_index()
    labels = [window_label(w) for w in agg.index]
    colors = [diag.COLOR_OUTLIER if v > 0 else diag.COLOR_LINE for v in agg.values]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, agg.values, color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("3-month test window")
    ax.set_ylabel("Mean Pearson residual")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def early_late_residuals(scored, early_years=(2022,), late_years=(2024,)):
    scored = scored.copy()
    scored["year"] = 2018 + scored["window_idx"] // 4
    early = scored[scored["year"].isin(early_years)]
    late = scored[scored["year"].isin(late_years)]
    return {
        "early": {"mean_resid": early["resid"].mean(), "mean_pearson": early["pearson_resid"].mean()},
        "late": {"mean_resid": late["resid"].mean(), "mean_pearson": late["pearson_resid"].mean()},
    }


def run_fixed_fit(df, train_mask, test_mask, month_idx, window_idx):
    print("\n" + "=" * 78)
    print("STEP 1 — FIT EXPECTED-COUNT MODEL ON TRAINING PERIOD (2018-2020)")
    print("=" * 78)

    train_df = df[train_mask].copy()
    train_windows = sorted(train_df["window_idx"].unique())
    train_cells = train_df[FEATURES].drop_duplicates()
    train_scaffold = build_scaffold(train_df, train_cells, train_windows)
    bundle = fit_expected_count_model(train_scaffold)

    train_levels = {c: set(train_df[c].unique()) for c in FEATURES}

    print("\n" + "=" * 78)
    print("STEP 2 — SCORE TEST PERIOD (fixed fit, no refit)")
    print("=" * 78)

    test_df = df[test_mask].copy()
    test_windows = sorted(test_df["window_idx"].unique())
    scored = score_cells(bundle, train_levels, test_df, test_windows)

    out_cols = FEATURES + ["window_idx", "window_label", "count", "expected_nb", "expected_pois", "resid", "pearson_resid"]
    out_path = config.REPORTS_DIR / "expected_counts_test.csv"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scored[out_cols].rename(columns={"count": "observed"}).to_csv(out_path, index=False)
    print(f"\nSaved (cell, window, observed, expected) artifact: {out_path} ({len(scored)} rows)")
    print(f"Total observed = {scored['count'].sum()}, total expected (NB) = {scored['expected_nb'].sum():.1f}")

    return bundle, train_levels, scored


def run_staleness_diagnostic(scored_fixed):
    print("\n" + "=" * 78)
    print("STEP 3 — STALENESS / RESIDUAL DIAGNOSTIC")
    print("=" * 78)

    plot_observed_vs_expected(
        scored_fixed, "observed_vs_expected_fixed_fit.png",
        "Observed vs. expected claims, fixed fit (trained 2018-2020)",
    )
    plot_residuals(
        scored_fixed, "residuals_fixed_fit.png",
        "Mean Pearson residual by test window, fixed fit",
    )

    fixed_el = early_late_residuals(scored_fixed)
    print(f"\nFixed fit — early (2022): mean resid = {fixed_el['early']['mean_resid']:+.3f}, "
          f"mean Pearson resid = {fixed_el['early']['mean_pearson']:+.3f}")
    print(f"Fixed fit — late  (2024): mean resid = {fixed_el['late']['mean_resid']:+.3f}, "
          f"mean Pearson resid = {fixed_el['late']['mean_pearson']:+.3f}")
    if fixed_el["late"]["mean_pearson"] > fixed_el["early"]["mean_pearson"] + 0.2:
        print(
            "-> Residuals fan out positive over the test period: the fixed fit "
            "increasingly UNDER-predicts late windows. This is the staleness signal."
        )

    return fixed_el


def run_expanding_refit(df):
    print("\nRobustness comparison (SECONDARY, not the main pipeline): expanding-window refit")
    print("Retrains at each test-year boundary; 2021 stays excluded throughout for consistency.")

    refit_scored_parts = []
    for stage in REFIT_STAGES:
        stage_train_mask = df["claim_date"].dt.year.isin(stage["train_years"])
        stage_train_df = df[stage_train_mask].copy()
        stage_train_windows = sorted(stage_train_df["window_idx"].unique())
        stage_train_cells = stage_train_df[FEATURES].drop_duplicates()
        stage_scaffold = build_scaffold(stage_train_df, stage_train_cells, stage_train_windows)
        stage_bundle = fit_expected_count_model(stage_scaffold, verbose=False)
        stage_train_levels = {c: set(stage_train_df[c].unique()) for c in FEATURES}

        predict_mask = df["claim_date"].dt.year == stage["predict_year"]
        predict_df = df[predict_mask].copy()
        predict_windows = sorted(predict_df["window_idx"].unique())
        scored_stage = score_cells(stage_bundle, stage_train_levels, predict_df, predict_windows, verbose=False)
        refit_scored_parts.append(scored_stage)
        print(
            f"  refit for {stage['predict_year']} (train={stage['train_years']}): "
            f"observed={scored_stage['count'].sum()}, expected={scored_stage['expected_nb'].sum():.1f}"
        )

    refit_scored = pd.concat(refit_scored_parts, ignore_index=True)
    refit_el = early_late_residuals(refit_scored)
    print(f"\nRefit — early (2022): mean resid = {refit_el['early']['mean_resid']:+.3f}, "
          f"mean Pearson resid = {refit_el['early']['mean_pearson']:+.3f}")
    print(f"Refit — late  (2024): mean resid = {refit_el['late']['mean_resid']:+.3f}, "
          f"mean Pearson resid = {refit_el['late']['mean_pearson']:+.3f}")

    plot_residuals(
        refit_scored, "residuals_expanding_refit.png",
        "Mean Pearson residual by test window, expanding-window refit (secondary)",
    )
    return refit_scored, refit_el


def report_occupancy_floor(scored_fixed):
    print("\n" + "=" * 78)
    print("STEP 4 — PER-WINDOW OCCUPANCY FLOOR")
    print("=" * 78)

    populated = scored_fixed[scored_fixed["count"] > 0]
    per_window_mean = populated.groupby("window_idx")["count"].mean().sort_index()
    per_window_mean.index = [window_label(w) for w in per_window_mean.index]

    print(f"\nMean claims per populated cell, per 3-month test window:")
    print(per_window_mean.to_string())
    worst_window = per_window_mean.idxmin()
    worst_value = per_window_mean.min()
    print(f"\nMinimum: {worst_value:.3f} ({worst_window}) vs. whole-period mean {FROZEN_DESIGN_REFERENCE['mean_per_cell']:.3f}")

    below_warn = per_window_mean[per_window_mean < OCCUPANCY_WARN_THRESHOLD]
    below_critical = per_window_mean[per_window_mean < OCCUPANCY_CRITICAL_THRESHOLD]
    if len(below_warn):
        print(f"\nWindows below the {OCCUPANCY_WARN_THRESHOLD:.1f} usability bar: {', '.join(below_warn.index)}")
    if len(below_critical):
        print(f"Windows below the {OCCUPANCY_CRITICAL_THRESHOLD:.1f} critical bar: {', '.join(below_critical.index)}")
    if not len(below_warn):
        print(f"\nNo test window falls below {OCCUPANCY_WARN_THRESHOLD:.1f} mean/cell.")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [
        diag.COLOR_OUTLIER if v < OCCUPANCY_CRITICAL_THRESHOLD
        else "#E8A33D" if v < OCCUPANCY_WARN_THRESHOLD
        else diag.COLOR_LINE
        for v in per_window_mean.values
    ]
    ax.bar(per_window_mean.index, per_window_mean.values, color=colors)
    ax.axhline(OCCUPANCY_WARN_THRESHOLD, color="#E8A33D", linewidth=1.5, linestyle="--", label=f">{OCCUPANCY_WARN_THRESHOLD:.0f} usable")
    ax.axhline(OCCUPANCY_CRITICAL_THRESHOLD, color=diag.COLOR_OUTLIER, linewidth=1.5, linestyle=":", label=f"{OCCUPANCY_CRITICAL_THRESHOLD:.1f} critical")
    ax.set_title("Mean claims per populated cell, per test window")
    ax.set_xlabel("3-month test window")
    ax.set_ylabel("Mean claims / populated cell")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out_path = config.REPORTS_DIR / "occupancy_floor_test.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")

    return per_window_mean, below_warn, below_critical


def print_summary(bundle, fixed_el, refit_el, per_window_mean, below_warn, below_critical):
    print("\n" + "=" * 78)
    print("STEP 5 — SUMMARY")
    print("=" * 78)

    print("\n2021 handling: BUFFER, excluded from both train and test.")
    print(f"NB dispersion (alpha): {bundle['alpha']:.4f}")
    print(f"Overdispersion test (Cameron-Trivedi, robust SE): p = {bundle['ct_pvalue']:.2e} "
          f"({'confirms' if bundle['ct_pvalue'] < 0.05 else 'does NOT confirm'} overdispersion)")
    print(f"Pearson chi2/df (Poisson fit): {bundle['pearson_ratio']:.2f}")
    print(f"Fitted trend slope: {bundle['pois'].params['window_idx']:+.4f} per window "
          f"({bundle['pois'].params['window_idx']*4:+.4f} annualized, log scale) — "
          f"{'DECLINING' if bundle['pois'].params['window_idx'] < 0 else 'RISING'} during training.")

    print("\nEarly (2022) vs late (2024) mean Pearson residual:")
    print(f"  {'':10s}{'fixed fit':>14s}{'expanding refit':>18s}")
    print(f"  {'early':10s}{fixed_el['early']['mean_pearson']:14.3f}{refit_el['early']['mean_pearson']:18.3f}")
    print(f"  {'late':10s}{fixed_el['late']['mean_pearson']:14.3f}{refit_el['late']['mean_pearson']:18.3f}")

    fixed_drift = fixed_el["late"]["mean_pearson"] - fixed_el["early"]["mean_pearson"]
    refit_drift = refit_el["late"]["mean_pearson"] - refit_el["early"]["mean_pearson"]
    print(f"\n  Fixed-fit drift (late - early): {fixed_drift:+.3f}")
    print(f"  Refit drift (late - early):     {refit_drift:+.3f}")
    if refit_drift < fixed_drift * 0.7:
        print("  -> Refitting materially reduces staleness drift.")
    else:
        print("  -> Refitting does not materially reduce staleness drift; the trend/seasonality "
              "terms alone can't keep up with the 2023-24 climb even with fresher data.")

    print(f"\nPer-window occupancy: minimum = {per_window_mean.min():.3f} ({per_window_mean.idxmin()}), "
          f"whole-period reference = {FROZEN_DESIGN_REFERENCE['mean_per_cell']:.3f}")
    if len(below_critical):
        print(f"  CRITICAL (<{OCCUPANCY_CRITICAL_THRESHOLD}): {', '.join(below_critical.index)}")
    if len(below_warn):
        print(f"  Below {OCCUPANCY_WARN_THRESHOLD:.0f} bar: {', '.join(below_warn.index)}")

    print("\nOverall assessment:")
    if fixed_el["late"]["mean_pearson"] > 0.5 or len(below_critical):
        print(
            "  The fixed-fit primary design shows a real problem before scanning: "
            f"{'late-window residuals fan out strongly positive (staleness)' if fixed_el['late']['mean_pearson'] > 0.5 else ''}"
            f"{' and ' if fixed_el['late']['mean_pearson'] > 0.5 and len(below_critical) else ''}"
            f"{'early/low-volume windows sit at or below the critical occupancy floor' if len(below_critical) else ''}. "
            "MDSS run on the raw fixed-fit expected counts will likely over-flag the 2023-24 "
            "back-end as 'emerging' when much of that signal is simply model staleness, and/or "
            "under-power the low-occupancy early windows. Consider: refreshing the trend term, "
            "shortening the scan's forecast horizon, or treating early-2022 flags with reduced "
            "confidence given thin occupancy."
        )
    else:
        print("  No major red flags: residuals stay reasonably centered and occupancy stays above the floor.")


def main():
    processed_md5_before = file_md5(config.PROCESSED_DATA_PATH)
    active_md5_before = file_md5(config.ACTIVE_DATA_PATH)

    df = diag.load_data(config.ACTIVE_DATA_PATH)
    for c in FEATURES:
        df[c] = df[c].astype(str)

    train_mask, test_mask, buffer_mask = resolve_train_test_split(df)

    month_idx, window_idx = diag.build_time_indices(df)
    df["window_idx"] = window_idx
    df["quarter"] = (window_idx % 4).astype(int)
    sanity_check_cell_grid(df, month_idx, window_idx)

    bundle, train_levels, scored_fixed = run_fixed_fit(df, train_mask, test_mask, month_idx, window_idx)
    fixed_el = run_staleness_diagnostic(scored_fixed)
    refit_scored, refit_el = run_expanding_refit(df)
    per_window_mean, below_warn, below_critical = report_occupancy_floor(scored_fixed)
    print_summary(bundle, fixed_el, refit_el, per_window_mean, below_warn, below_critical)

    processed_md5_after = file_md5(config.PROCESSED_DATA_PATH)
    active_md5_after = file_md5(config.ACTIVE_DATA_PATH)
    print("\n" + "=" * 78)
    print(f"Source unchanged — processed_data.csv: {processed_md5_before == processed_md5_after} "
          f"(MD5 {processed_md5_after})")
    print(f"Source unchanged — {config.ACTIVE_DATA_PATH.name}: {active_md5_before == active_md5_after} "
          f"(MD5 {active_md5_after})")


if __name__ == "__main__":
    main()
