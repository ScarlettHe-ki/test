"""Reverse trade-off from 02_subclass_coarsening_grid.py: drop sub_class (78
values) entirely, keep geography, and confirm which geography granularity
clears the 3-month >3 mean-claims/cell bar without it.

Read-only on config.PROCESSED_DATA_PATH and config.CLEANED_REGION_DATA_PATH
(writes no data files). Reuses cell-counting helpers from
01_sparsity_dispersion_check.py.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

BASE_SCAN_COLS = [
    "peril_type", "syndicate", "group_class",
    "placing_basis_group", "leader_status", "new_renewal", "risk_code",
]
GEO_VARIANTS = {"none": None, "4-region": "loss_census_region", "9-division": "loss_census_division"}

# Recorded from 02_subclass_coarsening_grid.py's Step 5 conclusion (sub_class
# in at 78 values, no geography) for the before/after comparison in Step 5.
PREVIOUS_DESIGN_REFERENCE = {
    "label": "base 7 + sub_class (78), no geography",
    "n_features": 8,
    "3-month": {"mean_per_cell": 3.113, "vmr": 14.044},
}


def define_feature_set(df):
    print("=" * 78)
    print("STEP 1 — DEFINE CANDIDATE FEATURE SETS")
    print("=" * 78)

    print(f"\nBase {len(BASE_SCAN_COLS)} features (sub_class removed):")
    for c in BASE_SCAN_COLS:
        print(f"  {c:24s} {df[c].nunique():5d} distinct values")

    sub_class_excluded = "sub_class" not in BASE_SCAN_COLS
    print(f"\nsub_class excluded from scan-feature set: {sub_class_excluded}")
    assert sub_class_excluded, "sub_class leaked back into BASE_SCAN_COLS"


def evaluate_combo(df, geo_col, month_idx, window_idx):
    cols = BASE_SCAN_COLS + ([geo_col] if geo_col else [])
    monthly_counts = diag.cell_counts(df, cols, month_idx)
    window_counts = diag.cell_counts(df, cols, window_idx)
    monthly_dist = diag.cell_distribution(monthly_counts)
    window_dist = diag.cell_distribution(window_counts)
    return {
        "cols": cols,
        "geo_cardinality": df[geo_col].nunique() if geo_col else 1,
        "monthly": {**monthly_dist, "vmr": diag._vmr(monthly_counts)},
        "3-month": {**window_dist, "vmr": diag._vmr(window_counts)},
        "usable_monthly": monthly_dist["mean_per_cell"] > diag.MEAN_PER_CELL_USABLE_THRESHOLD,
        "usable_3month": window_dist["mean_per_cell"] > diag.MEAN_PER_CELL_USABLE_THRESHOLD,
    }


def run_grid(df, month_idx, window_idx):
    print("\n" + "=" * 78)
    print("STEP 2 — SPARSITY GRID: 1-month vs 3-month, by geography level")
    print("=" * 78)

    grid = {}
    for geo_name, geo_col in GEO_VARIANTS.items():
        grid[geo_name] = evaluate_combo(df, geo_col, month_idx, window_idx)

    labels = {
        "none": "(a) base 7, no geography",
        "4-region": "(b) base 7 + loss_census_region (4)",
        "9-division": "(c) base 7 + loss_census_division (9)",
    }
    print(f"\n{'feature set':38s}{'1-month mean/cell':>20s}{'verdict':>9s}{'3-month mean/cell':>20s}{'verdict':>9s}")
    for geo_name, label in labels.items():
        r = grid[geo_name]
        print(
            f"{label:38s}"
            f"{r['monthly']['mean_per_cell']:20.3f}{'PASS' if r['usable_monthly'] else 'fail':>9s}"
            f"{r['3-month']['mean_per_cell']:20.3f}{'PASS' if r['usable_3month'] else 'fail':>9s}"
        )

    return grid


def print_recommendation(grid):
    print("\n" + "=" * 78)
    print("STEP 3 — RECOMMENDATION")
    print("=" * 78)

    threshold = diag.MEAN_PER_CELL_USABLE_THRESHOLD
    for geo_name in ("9-division", "4-region"):
        r = grid[geo_name]
        mean = r["3-month"]["mean_per_cell"]
        margin = mean - threshold
        verdict = "PASSES" if r["usable_3month"] else "fails"
        print(
            f"\n{geo_name}: 3-month mean/cell = {mean:.3f}, {verdict} the >{threshold:.0f} bar "
            f"(margin {margin:+.3f}, {margin / threshold:+.1%})."
        )

    if grid["9-division"]["usable_3month"]:
        chosen = "9-division"
        print(
            "\n-> 9-division PASSES: the finer geography resolution is available. "
            "Recommending loss_census_division for the finer research-relevant "
            "granularity."
        )
    elif grid["4-region"]["usable_3month"]:
        chosen = "4-region"
        print(
            "\n-> 9-division fails; falling back to 4-region, which passes."
        )
    else:
        chosen = None
        print(
            "\n-> Even 4-region fails with sub_class removed. The base features "
            "alone are the limit — geography cannot be supported at any tested "
            "granularity without further redesign."
        )

    return chosen


def print_dispersion(df, grid, chosen, month_idx, window_idx):
    print("\n" + "=" * 78)
    print("STEP 4 — DISPERSION for the recommended design")
    print("=" * 78)

    if chosen is None:
        print("\nNo geography-inclusive design passed; nothing to recompute.")
        r = grid["none"]
        label = "base 7, no geography"
    else:
        r = grid[chosen]
        label = f"base 7 + {chosen}"

    print(f"\n{label}:")
    print(f"  monthly  VMR = {r['monthly']['vmr']:.3f}")
    print(f"  3-month  VMR = {r['3-month']['vmr']:.3f}")

    monthly_totals = df.groupby(df["claim_date"].dt.to_period("M")).size().sort_index()
    total_vmr = monthly_totals.var() / monthly_totals.mean()
    print(f"\nMonthly TOTAL claim count series: mean={monthly_totals.mean():.1f}, "
          f"var={monthly_totals.var():.1f}, variance/mean={total_vmr:.2f}")

    z = (monthly_totals - monthly_totals.mean()) / monthly_totals.std()
    outliers = monthly_totals[z.abs() > 2]
    dist_rec = "negative binomial" if r["3-month"]["vmr"] > 1.5 or total_vmr > 1.5 else "Poisson"
    print(f"\nNull distribution: {dist_rec.upper()} "
          f"(cell-level VMR {r['3-month']['vmr']:.2f}, monthly total VMR {total_vmr:.2f}, both >> 1).")

    return monthly_totals, z, outliers, total_vmr, dist_rec


PLOT_LABELS = {"none": "base7 (no geo)", "4-region": "base7 + 4-region", "9-division": "base7 + 9-division"}


def plot_results(df, grid, monthly_totals, z):
    sparsity_results_for_plot = {
        PLOT_LABELS[k]: {"monthly": v["monthly"], "3-month": v["3-month"]}
        for k, v in grid.items()
    }
    diag.plot_cell_distributions(sparsity_results_for_plot, filename="cell_count_distribution_geo_kept.png")
    diag.plot_monthly_totals(monthly_totals, z, filename="monthly_total_claims_geo_kept.png")


def print_final_decision(grid, chosen, dist_rec, outliers):
    print("\n" + "=" * 78)
    print("STEP 5 — FINAL DECISION")
    print("=" * 78)

    if chosen:
        geo_col = GEO_VARIANTS[chosen]
        final_cols = BASE_SCAN_COLS + [geo_col]
        r = grid[chosen]
        window = "3-MONTH"
    else:
        final_cols = BASE_SCAN_COLS
        r = grid["none"]
        window = "3-MONTH"

    print(f"\nFrozen scan-feature set ({len(final_cols)} features):")
    for c in final_cols:
        print(f"    {c}")

    print(f"\nDetection window: {window}")
    print(f"  mean/cell = {r['3-month']['mean_per_cell']:.3f}")
    print(f"\nNull distribution: {dist_rec.upper()}")
    if len(outliers):
        print(f"Candidate catastrophe/event months: {', '.join(str(p) for p in outliers.index)}")

    print(
        f"\nBefore/after vs. previous design ({PREVIOUS_DESIGN_REFERENCE['label']}, "
        f"{PREVIOUS_DESIGN_REFERENCE['n_features']} features, "
        f"3-month mean/cell={PREVIOUS_DESIGN_REFERENCE['3-month']['mean_per_cell']:.3f}):"
    )
    if chosen:
        print(
            f"  Gave up sub_class resolution (78 -> 0 categories) to gain "
            f"{chosen} geography ({r['geo_cardinality']} categories); "
            f"3-month mean/cell {PREVIOUS_DESIGN_REFERENCE['3-month']['mean_per_cell']:.3f} -> "
            f"{r['3-month']['mean_per_cell']:.3f}."
        )
    else:
        print("  Gave up sub_class resolution but geography still could not be supported.")


def main():
    processed_md5_before = file_md5(config.PROCESSED_DATA_PATH)
    cleaned_md5_before = file_md5(config.CLEANED_REGION_DATA_PATH)

    df = diag.load_data(config.CLEANED_REGION_DATA_PATH)

    define_feature_set(df)

    month_idx, window_idx = diag.build_time_indices(df)
    grid = run_grid(df, month_idx, window_idx)
    chosen = print_recommendation(grid)
    monthly_totals, z, outliers, total_vmr, dist_rec = print_dispersion(df, grid, chosen, month_idx, window_idx)
    plot_results(df, grid, monthly_totals, z)
    print_final_decision(grid, chosen, dist_rec, outliers)

    processed_md5_after = file_md5(config.PROCESSED_DATA_PATH)
    cleaned_md5_after = file_md5(config.CLEANED_REGION_DATA_PATH)
    print("\n" + "=" * 78)
    print(f"Source unchanged — processed_data.csv: {processed_md5_before == processed_md5_after} "
          f"(MD5 {processed_md5_after})")
    print(f"Source unchanged — processed_data_region_clean.csv: {cleaned_md5_before == cleaned_md5_after} "
          f"(MD5 {cleaned_md5_after})")


if __name__ == "__main__":
    main()
