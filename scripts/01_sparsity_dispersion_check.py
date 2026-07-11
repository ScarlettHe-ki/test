"""Pre-flight diagnostics for subgroup-scan design: sparsity and dispersion.

Read-only: loads data/processed_data.csv, prints a report, and saves plots to
REPORTS_DIR. Does not modify anything under data/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# Columns that are identifiers, the time field itself, or constant/label
# columns rather than independent scan features. Determined by inspecting
# the actual data (see STEP 1 output): event_id is a per-row key,
# claim_month/claim_date carry the time information, group_division has a
# single value across all rows, and is_group_cat_event is a derived
# catastrophe-cluster flag rather than an independent claim/policy attribute.
NON_SCAN_COLUMNS = {
    "event_id",
    "claim_month",
    "claim_date",
    "group_division",
    "is_group_cat_event",
}
# The geography scan feature has been through two revisions of the pipeline:
# loss_city (2253 values, dropped for sparsity) -> loss_region (coarser).
# Detect whichever is present rather than hardcoding one, so the script keeps
# working if the geography column changes again.
GEO_FEATURE_CANDIDATES = ("loss_region", "loss_city")
HIGH_CARDINALITY_FRACTION_THRESHOLD = 0.05  # feature cardinality / row count
MEAN_PER_CELL_USABLE_THRESHOLD = 3.0  # rule-of-thumb floor for a stable scan cell

# Recorded from the loss_city run (2026-07-11) for the before/after comparison
# in Step 4. That run's source data has since been overwritten, so these are
# fixed reference numbers rather than something recomputed live.
OLD_RUN_REFERENCE = {
    "label": "loss_city run, excl. loss_city scenario (2026-07-11)",
    "monthly": {"frac_1": 0.640, "frac_2": 0.163, "mean_per_cell": 2.610},
    "3-month": {"frac_1": 0.566, "frac_2": 0.176, "mean_per_cell": 3.117},
}

# Color choices for the diagnostic plots (single sequential hue for
# magnitude, one status-red reserved for flagged outlier months).
COLOR_LINE = "#3B6FA0"
COLOR_OUTLIER = "#C0392B"
COLOR_BUCKET = {"1": "#C6D9EC", "2": "#6D9BC3", "3+": "#2C5A85"}


def load_data(path):
    df = pd.read_csv(path)
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    return df


def report_shape_and_dtypes(df):
    print("=" * 78)
    print("STEP 1 — INSPECT AND IDENTIFY")
    print("=" * 78)
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("Dtypes:")
    print(df.dtypes.to_string())
    print("\nPer-column unique-value counts:")
    print(df.nunique().to_string())
    n_nulls = df.isnull().sum()
    if n_nulls.any():
        print("\nColumns with nulls:")
        print(n_nulls[n_nulls > 0].to_string())
    else:
        print("\nNo nulls in any column.")


def report_date_column(df):
    date_min, date_max = df["claim_date"].min(), df["claim_date"].max()
    n_months = df["claim_date"].dt.to_period("M").nunique()
    month_consistent = (
        df["claim_date"].dt.strftime("%Y-%m") == df["claim_month"]
    ).all()
    print("\nDate column: 'claim_date' (daily); 'claim_month' is a redundant")
    print("YYYY-MM bucket derived from it "
          f"(matches claim_date in all rows: {month_consistent}).")
    print(f"Date range: {date_min.date()} -> {date_max.date()}")
    print(f"Span: {n_months} distinct months")
    return n_months


def detect_geo_feature(df):
    for candidate in GEO_FEATURE_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No geography column found among {GEO_FEATURE_CANDIDATES}")


def describe_state_code_quality(series):
    """For loss_region: how many values are clean 2-letter state codes vs
    multi-region combos ('TX | AL') vs other non-conforming entries (full
    state names, city names, malformed codes)."""
    values = series.value_counts()
    is_clean = values.index.str.match(r"^[A-Z]{2}$")
    is_multi = values.index.str.contains(r"\|")
    is_other = ~is_clean & ~is_multi

    print(
        f"  Value-format check on '{series.name}': "
        f"{is_clean.sum()} clean 2-letter codes, {is_multi.sum()} multi-region "
        f"combos (e.g. 'TX | AL'), {is_other.sum()} other non-conforming values "
        "(full state names, city names, malformed codes)."
    )
    if is_other.sum():
        other_rows = values[is_other].sum()
        print(
            f"  Non-conforming values cover {other_rows} rows "
            f"({other_rows / series.notna().sum():.2%}) — data-quality tail, "
            "not a scan-relevant category split. Examples: "
            f"{', '.join(values[is_other].index[:6])}"
        )


def identify_categorical_features(df):
    scan_cols = [c for c in df.columns if c not in NON_SCAN_COLUMNS]
    cardinalities = {c: df[c].nunique() for c in scan_cols}
    geo_feature = detect_geo_feature(df)

    print("\nCategorical scan features (claim-level + policy-level):")
    for c, k in sorted(cardinalities.items(), key=lambda kv: kv[1]):
        flag = " <-- geography feature" if c == geo_feature else ""
        print(f"  {c:24s} {k:5d} distinct values{flag}")

    n_rows = len(df)
    geo_card = cardinalities[geo_feature]
    geo_counts = df[geo_feature].value_counts()
    singleton_share = (geo_counts == 1).sum() / geo_card
    top15_share = geo_counts.head(15).sum() / n_rows
    geo_ratio = geo_card / n_rows
    geo_flagged = geo_ratio > HIGH_CARDINALITY_FRACTION_THRESHOLD

    print(
        f"\n'{geo_feature}' has {geo_card} distinct values across "
        f"{n_rows} rows ({geo_ratio:.1%} of row count)."
    )
    print(
        f"  {(geo_counts == 1).sum()}/{geo_card} values ({singleton_share:.1%}) "
        "occur exactly once in the whole 7-year dataset."
    )
    print(f"  Top 15 values cover only {top15_share:.1%} of rows (long tail).")
    print(
        f"  Cardinality/rows = {geo_ratio:.4f} "
        f"(threshold {HIGH_CARDINALITY_FRACTION_THRESHOLD}) -> "
        f"{'FLAGGED as a coarsening/dropping candidate on cardinality alone.' if geo_flagged else 'does not trip the cardinality heuristic on its own; see Step 2 for the effective sparsity test.'}"
    )
    if geo_feature == "loss_region":
        describe_state_code_quality(df[geo_feature])

    return scan_cols, cardinalities, geo_feature, geo_flagged


def build_time_indices(df):
    """Month index (0-based from first month) and its //3 3-month bucket."""
    start = df["claim_date"].min().to_period("M")
    month_idx = (df["claim_date"].dt.to_period("M") - start).apply(lambda o: o.n)
    window_idx = month_idx // 3
    return month_idx, window_idx


def cell_counts(df, features, time_idx):
    key = df[features].astype(str).agg("|".join, axis=1)
    return df.groupby([key, time_idx]).size()


def cell_distribution(counts):
    n_cells = len(counts)
    frac1 = (counts == 1).mean()
    frac2 = (counts == 2).mean()
    frac3p = (counts >= 3).mean()
    return {
        "n_populated_cells": n_cells,
        "frac_1": frac1,
        "frac_2": frac2,
        "frac_3plus": frac3p,
        "mean_per_cell": counts.mean(),
    }


def report_sparsity(df, scan_cols, cardinalities, geo_feature, month_idx, window_idx, n_months):
    print("\n" + "=" * 78)
    print("STEP 2 — SPARSITY CHECK")
    print("=" * 78)

    claims_per_month = len(df) / n_months
    print(f"\nClaims per month overall: {claims_per_month:.1f} "
          f"({len(df)} claims / {n_months} months)")

    excl_geo_cols = [c for c in scan_cols if c != geo_feature]
    n_windows = window_idx.nunique()

    primary_key = f"incl. {geo_feature}"
    excl_key = f"excl. {geo_feature}"
    scenarios = {primary_key: scan_cols, excl_key: excl_geo_cols}
    roles = {primary_key: "PRIMARY", excl_key: "SECONDARY (comparison)"}

    results = {}
    for name, cols in scenarios.items():
        cross_product = int(np.prod([cardinalities[c] for c in cols]))
        print(f"\n--- {roles[name]}: ALL features {name} ({len(cols)} features) ---")
        print(f"  Theoretical cross-product of feature cardinalities: {cross_product:,}")
        print(f"  x {n_months} months = {cross_product * n_months:,} theoretical monthly cells")
        print(f"  x {n_windows} 3-month windows = {cross_product * n_windows:,} theoretical 3-month cells")

        monthly_counts = cell_counts(df, cols, month_idx)
        window_counts = cell_counts(df, cols, window_idx)
        monthly_dist = cell_distribution(monthly_counts)
        window_dist = cell_distribution(window_counts)
        results[name] = {"monthly": monthly_dist, "3-month": window_dist}

        print(f"  {'':18s}{'monthly':>14s}{'3-month':>14s}")
        for key, label in [
            ("n_populated_cells", "populated cells"),
            ("frac_1", "frac count==1"),
            ("frac_2", "frac count==2"),
            ("frac_3plus", "frac count>=3"),
            ("mean_per_cell", "mean/cell"),
        ]:
            m, w = monthly_dist[key], window_dist[key]
            if key == "n_populated_cells":
                print(f"  {label:18s}{m:14,d}{w:14,d}")
            else:
                print(f"  {label:18s}{m:14.3f}{w:14.3f}")

    primary_3month_mean = results[primary_key]["3-month"]["mean_per_cell"]
    usable = primary_3month_mean > MEAN_PER_CELL_USABLE_THRESHOLD
    print(
        f"\nKey question: with {geo_feature} included, does the 3-month mean/cell "
        f"stay usable (> {MEAN_PER_CELL_USABLE_THRESHOLD:.0f})? "
        f"{'YES' if usable else 'NO'} — {primary_3month_mean:.2f}/cell "
        f"({'above' if usable else 'below'} threshold; {geo_feature}'s "
        f"{cardinalities[geo_feature]} categories still fragment cells "
        f"{'without pulling it back to singleton-dominated.' if usable else 'and pull it back toward singleton-dominated.'})"
    )

    return results, excl_geo_cols, primary_key


def report_dispersion(df, scan_cols, excl_geo_cols, geo_feature, month_idx, window_idx):
    print("\n" + "=" * 78)
    print("STEP 3 — DISPERSION CHECK")
    print("=" * 78)

    print(f"\n{'':32s}{'monthly VMR':>14s}{'3-month VMR':>14s}")
    vmr_table = {}
    for name, cols in [
        (f"ALL features (incl. {geo_feature})", scan_cols),
        (f"ALL features excl. {geo_feature}", excl_geo_cols),
    ]:
        monthly_vmr = _vmr(cell_counts(df, cols, month_idx))
        window_vmr = _vmr(cell_counts(df, cols, window_idx))
        vmr_table[name] = {"monthly": monthly_vmr, "3-month": window_vmr}
        print(f"  {name:30s}{monthly_vmr:14.3f}{window_vmr:14.3f}")

    monthly_totals = df.groupby(df["claim_date"].dt.to_period("M")).size().sort_index()
    total_vmr = monthly_totals.var() / monthly_totals.mean()
    print(f"\nMonthly TOTAL claim count series: mean={monthly_totals.mean():.1f}, "
          f"var={monthly_totals.var():.1f}, variance/mean={total_vmr:.2f}")

    z = (monthly_totals - monthly_totals.mean()) / monthly_totals.std()
    outliers = monthly_totals[z.abs() > 2]
    if len(outliers):
        print(f"Outlier months (|z| > 2, candidate catastrophe/event months):")
        for period, count in outliers.items():
            print(f"  {period}: {count} claims (z={z[period]:.2f})")
    else:
        print("No months exceed |z| > 2.")

    plot_monthly_totals(monthly_totals, z)
    return vmr_table, total_vmr, outliers


def _vmr(counts):
    return counts.var() / counts.mean()


def plot_monthly_totals(monthly_totals, z, filename="monthly_total_claims.png"):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = monthly_totals.index.to_timestamp()
    ax.plot(x, monthly_totals.values, color=COLOR_LINE, linewidth=2, zorder=2)
    outlier_mask = z.abs() > 2
    ax.scatter(
        x[outlier_mask],
        monthly_totals.values[outlier_mask],
        color=COLOR_OUTLIER,
        s=40,
        zorder=3,
        label="|z| > 2 (candidate event month)",
    )
    ax.set_title("Monthly total claim counts")
    ax.set_xlabel("Month")
    ax.set_ylabel("Claims")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


def plot_cell_distributions(sparsity_results, filename="cell_count_distribution.png"):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scenario_names = list(sparsity_results.keys())
    windows = ["monthly", "3-month"]
    bar_labels = [f"{s}\n({w})" for s in scenario_names for w in windows]
    frac1 = [sparsity_results[s][w]["frac_1"] for s in scenario_names for w in windows]
    frac2 = [sparsity_results[s][w]["frac_2"] for s in scenario_names for w in windows]
    frac3p = [sparsity_results[s][w]["frac_3plus"] for s in scenario_names for w in windows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(bar_labels))
    ax.bar(x, frac1, color=COLOR_BUCKET["1"], label="count == 1")
    ax.bar(x, frac2, bottom=frac1, color=COLOR_BUCKET["2"], label="count == 2")
    bottom2 = np.array(frac1) + np.array(frac2)
    ax.bar(x, frac3p, bottom=bottom2, color=COLOR_BUCKET["3+"], label="count >= 3")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Fraction of populated cells")
    ax.set_title("Populated (feature-combo x time-window) cell count distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out_path = config.REPORTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def print_summary(
    sparsity_results, primary_key, scan_cols, cardinalities, geo_feature, geo_flagged,
    vmr_table, total_vmr, outliers, n_rows,
):
    print("\n" + "=" * 78)
    print("STEP 4 — SUMMARY / DECISION")
    print("=" * 78)

    primary_monthly = sparsity_results[primary_key]["monthly"]
    primary_window = sparsity_results[primary_key]["3-month"]

    if primary_monthly["frac_1"] + primary_monthly["frac_2"] > 0.75:
        window_rec = "3-month"
        window_reason = (
            f"with the final {len(scan_cols)}-feature set (incl. {geo_feature}), monthly cells are "
            f"{primary_monthly['frac_1']:.0%} singletons + {primary_monthly['frac_2']:.0%} pairs "
            f"(mean {primary_monthly['mean_per_cell']:.2f}/cell); 3-month windows raise the mean "
            f"to {primary_window['mean_per_cell']:.2f}/cell and cut the singleton share to "
            f"{primary_window['frac_1']:.0%}."
        )
    else:
        window_rec = "monthly"
        window_reason = "monthly cells are not overwhelmingly 0/1 with the final feature set."

    max_vmr = max(v for scen in vmr_table.values() for v in scen.values())
    if max_vmr > 1.5 or total_vmr > 1.5:
        dist_rec = "negative binomial"
        dist_reason = (
            f"cell-level variance/mean ratios range up to {max_vmr:.2f}, and the monthly "
            f"total series has variance/mean = {total_vmr:.2f} (both >> 1)."
        )
    else:
        dist_rec = "Poisson"
        dist_reason = "variance/mean ratios are close to 1 at both the cell and aggregate level."

    print(f"\nFinal scan-feature set ({len(scan_cols)} features):")
    for c in sorted(scan_cols, key=lambda c: cardinalities[c]):
        print(f"  {c:24s} {cardinalities[c]:5d} distinct values")

    print(f"\nRecommended detection window: {window_rec.upper()}")
    print(f"  Why: {window_reason}")
    print(f"\nRecommended null distribution: {dist_rec.upper()}")
    print(f"  Why: {dist_reason}")
    if len(outliers):
        months_str = ", ".join(str(p) for p in outliers.index)
        print(f"  Candidate catastrophe/event months to watch: {months_str}")

    usable_3month = primary_window["mean_per_cell"] > MEAN_PER_CELL_USABLE_THRESHOLD
    print(f"\n'{geo_feature}' coarsening check:")
    print(
        f"  Cardinality-ratio heuristic: "
        f"{'flagged' if geo_flagged else 'not flagged'} "
        f"({cardinalities[geo_feature]} values / {n_rows} rows)."
    )
    print(
        f"  Effective sparsity test (3-month mean/cell with {geo_feature} in): "
        f"{primary_window['mean_per_cell']:.2f} "
        f"({'passes' if usable_3month else 'fails'} the "
        f"> {MEAN_PER_CELL_USABLE_THRESHOLD:.0f} usability bar)."
    )
    if not usable_3month:
        print(
            f"  Recommendation: consider a coarser regional grouping than "
            f"'{geo_feature}' (e.g. ~4-10 census regions/divisions instead of "
            f"{cardinalities[geo_feature]} state-level values) if cell sparsity "
            "needs to improve further."
        )

    print("\nBefore/after comparison vs. the loss_city run:")
    print(f"  {'':38s}{'singleton %':>14s}{'mean/cell':>12s}")
    old_m, old_w = OLD_RUN_REFERENCE["monthly"], OLD_RUN_REFERENCE["3-month"]
    print(f"  OLD ({OLD_RUN_REFERENCE['label']})")
    print(f"    monthly (excl. loss_city){'':13s}{old_m['frac_1']:13.1%}{old_m['mean_per_cell']:12.2f}")
    print(f"    3-month (excl. loss_city){'':13s}{old_w['frac_1']:13.1%}{old_w['mean_per_cell']:12.2f}")
    print(f"  NEW (this run, incl. {geo_feature})")
    print(f"    monthly{'':31s}{primary_monthly['frac_1']:13.1%}{primary_monthly['mean_per_cell']:12.2f}")
    print(f"    3-month{'':31s}{primary_window['frac_1']:13.1%}{primary_window['mean_per_cell']:12.2f}")
    if primary_window["mean_per_cell"] >= old_w["mean_per_cell"]:
        verdict = "city -> region genuinely reduced sparsity at the 3-month window."
    else:
        verdict = (
            "city -> region did not fully fix sparsity: keeping a geography "
            "feature in at all (even coarse) still fragments cells more than "
            "dropping it did."
        )
    print(f"  Verdict: {verdict}")


def main():
    df = load_data(config.ACTIVE_DATA_PATH)

    report_shape_and_dtypes(df)
    n_months = report_date_column(df)
    scan_cols, cardinalities, geo_feature, geo_flagged = identify_categorical_features(df)

    month_idx, window_idx = build_time_indices(df)
    sparsity_results, excl_geo_cols, primary_key = report_sparsity(
        df, scan_cols, cardinalities, geo_feature, month_idx, window_idx, n_months
    )
    plot_cell_distributions(sparsity_results)

    vmr_table, total_vmr, outliers = report_dispersion(
        df, scan_cols, excl_geo_cols, geo_feature, month_idx, window_idx
    )

    print_summary(
        sparsity_results,
        primary_key,
        scan_cols,
        cardinalities,
        geo_feature,
        geo_flagged,
        vmr_table,
        total_vmr,
        outliers,
        len(df),
    )


if __name__ == "__main__":
    main()
