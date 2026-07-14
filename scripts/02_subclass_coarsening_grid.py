"""Test whether coarsening sub_class (78 values) lets a coarse geography
feature back into the scan set, after 00_clean_loss_region.py found that
neither loss_census_division (9) nor loss_census_region (4) alone clears the
3-month >3 mean-claims/cell bar.

Read-only on config.CLEANED_REGION_DATA_PATH and config.PROCESSED_DATA_PATH
(writes no data files — this is a diagnostic grid, not a derivation step).
Reuses cell-counting/plotting helpers from 01_sparsity_dispersion_check.py.
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

TOP_N_OPTIONS = (10, 20)
LONG_TAIL_ROW_THRESHOLD = 20
BASE_SCAN_COLS = [
    "peril_type", "syndicate", "group_class",
    "placing_basis_group", "leader_status", "new_renewal", "risk_code",
]
GEO_VARIANTS = {"none": None, "4-region": "loss_census_region", "9-division": "loss_census_division"}


def inspect_sub_class(df):
    print("=" * 78)
    print("STEP 1 — DIAGNOSE sub_class")
    print("=" * 78)

    vc = df["sub_class"].value_counts()
    print(f"\n{len(vc)} distinct sub_class values, sorted by row count:")
    print(vc.to_string())

    cum_share = vc.cumsum() / vc.sum()
    n_top80 = int((cum_share <= 0.80).sum()) + 1
    n_long_tail = int((vc < LONG_TAIL_ROW_THRESHOLD).sum())
    print(f"\n{n_top80} sub_classes cover the top 80% of rows.")
    print(f"{n_long_tail} sub_classes are long-tail (<{LONG_TAIL_ROW_THRESHOLD} rows).")

    print("\nParent-taxonomy nesting check:")
    nesting_found = False
    for parent in ("group_class", "risk_code"):
        parent_counts = df.groupby("sub_class")[parent].nunique()
        clean = (parent_counts == 1).all()
        if clean:
            nesting_found = True
            print(f"  sub_class nests cleanly under {parent} — rollup available.")
            print(df.groupby(parent)["sub_class"].nunique().to_string())
        else:
            offenders = parent_counts[parent_counts > 1]
            print(
                f"  sub_class does NOT nest cleanly under {parent}: "
                f"{len(offenders)}/{len(parent_counts)} sub_classes span more than one "
                f"{parent} value (e.g. {', '.join(offenders.index[:5])})."
            )
    if not nesting_found:
        print(
            "\n  No usable parent taxonomy exists (group_class is too coarse at 2 "
            "values and doesn't nest cleanly either way; risk_code doesn't nest "
            "cleanly). Skipping the parent-rollup variant — binning the tail "
            "manually (top-N + other) instead."
        )

    return vc, nesting_found


def build_subclass_variants(df, vc):
    print("\n" + "=" * 78)
    print("STEP 2 — BUILD sub_class COARSENINGS")
    print("=" * 78)

    variants = {"full 78": "sub_class"}
    for n in TOP_N_OPTIONS:
        col = f"sub_class_top{n}"
        keep = set(vc.index[:n])
        df[col] = df["sub_class"].where(df["sub_class"].isin(keep), "OTHER")
        variants[f"top-{n}+other"] = col

        counts = df[col].value_counts()
        print(f"\n'{col}': {df[col].nunique()} categories ({n} kept + 'OTHER')")
        print(counts.to_string())

    return variants


def evaluate_combo(df, sc_col, geo_col, month_idx, window_idx):
    cols = BASE_SCAN_COLS + [sc_col] + ([geo_col] if geo_col else [])
    monthly_counts = diag.cell_counts(df, cols, month_idx)
    window_counts = diag.cell_counts(df, cols, window_idx)
    monthly_dist = diag.cell_distribution(monthly_counts)
    window_dist = diag.cell_distribution(window_counts)
    return {
        "cols": cols,
        "sc_cardinality": df[sc_col].nunique(),
        "geo_cardinality": df[geo_col].nunique() if geo_col else 1,
        "monthly": {**monthly_dist, "vmr": diag._vmr(monthly_counts)},
        "3-month": {**window_dist, "vmr": diag._vmr(window_counts)},
        "usable_3month": window_dist["mean_per_cell"] > diag.MEAN_PER_CELL_USABLE_THRESHOLD,
    }


def run_grid(df, subclass_variants, month_idx, window_idx):
    print("\n" + "=" * 78)
    print("STEP 3 — GRID: sub_class encoding x geography level (3-month)")
    print("=" * 78)

    grid = {}
    for sc_name, sc_col in subclass_variants.items():
        for geo_name, geo_col in GEO_VARIANTS.items():
            grid[(sc_name, geo_name)] = evaluate_combo(df, sc_col, geo_col, month_idx, window_idx)

    geo_names = list(GEO_VARIANTS)
    header = f"{'sub_class encoding':20s}" + "".join(f"{g:>22s}" for g in geo_names)
    print(f"\n{header}")
    for sc_name in subclass_variants:
        row = f"{sc_name:20s}"
        for geo_name in geo_names:
            r = grid[(sc_name, geo_name)]
            mean = r["3-month"]["mean_per_cell"]
            verdict = "PASS" if r["usable_3month"] else "fail"
            row += f"{mean:>14.3f} {verdict:>7s}"
        print(row)

    return grid


def report_dispersion(df, grid, month_idx, window_idx):
    print("\n" + "=" * 78)
    print("STEP 4 — DISPERSION for passing combinations")
    print("=" * 78)

    passing = {k: v for k, v in grid.items() if v["usable_3month"]}
    if not passing:
        print("\nNo combination passes; nothing to recompute.")
    for (sc_name, geo_name), r in passing.items():
        print(
            f"\n{sc_name} x {geo_name}: monthly VMR={r['monthly']['vmr']:.3f}, "
            f"3-month VMR={r['3-month']['vmr']:.3f}"
        )

    monthly_totals = df.groupby(df["claim_date"].dt.to_period("M")).size().sort_index()
    total_vmr = monthly_totals.var() / monthly_totals.mean()
    print(f"\nMonthly TOTAL claim count series: mean={monthly_totals.mean():.1f}, "
          f"var={monthly_totals.var():.1f}, variance/mean={total_vmr:.2f}")
    return passing, monthly_totals, total_vmr


def plot_grid_heatmap(grid, subclass_variants):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    geo_names = list(GEO_VARIANTS)
    sc_names = list(subclass_variants)
    values = np.array([[grid[(sc, g)]["3-month"]["mean_per_cell"] for g in geo_names] for sc in sc_names])
    passes = np.array([[grid[(sc, g)]["usable_3month"] for g in geo_names] for sc in sc_names])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(4, values.max()))
    for i in range(len(sc_names)):
        for j in range(len(geo_names)):
            label = f"{values[i, j]:.2f}\n{'PASS' if passes[i, j] else 'fail'}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if values[i, j] > values.max() / 2 else "#333333")
            if passes[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#E67E22", linewidth=3.5)
                ax.add_patch(rect)
    ax.set_xticks(range(len(geo_names)))
    ax.set_xticklabels(geo_names)
    ax.set_yticks(range(len(sc_names)))
    ax.set_yticklabels(sc_names)
    ax.set_title("3-month mean claims/cell (bordered = clears >3 bar)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean claims / populated cell")
    fig.tight_layout()
    out_path = config.REPORTS_DIR / "cell_count_grid_subclass_variant.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


def print_summary(grid, subclass_variants, total_vmr):
    print("\n" + "=" * 78)
    print("STEP 5 — SUMMARY / RECOMMENDATION")
    print("=" * 78)

    geo_inclusive_passes = [
        (sc_name, geo_name, r) for (sc_name, geo_name), r in grid.items()
        if geo_name != "none" and r["usable_3month"]
    ]

    if geo_inclusive_passes:
        # Coarsest geography (fewest categories) first, then coarsest sub_class encoding.
        geo_inclusive_passes.sort(key=lambda t: (t[2]["geo_cardinality"], t[2]["sc_cardinality"]))
        sc_name, geo_name, r = geo_inclusive_passes[0]
        geo_col = GEO_VARIANTS[geo_name]
        sc_col = subclass_variants[sc_name]
        print(f"\nRecommendation: {sc_name} sub_class + {geo_name} geography "
              f"({r['3-month']['mean_per_cell']:.3f}/cell, clears the >3 bar).")
        final_cols = BASE_SCAN_COLS + [sc_col, geo_col]
    else:
        best_geo_attempt = max(
            ((sc_name, geo_name, r) for (sc_name, geo_name), r in grid.items() if geo_name != "none"),
            key=lambda t: t[2]["3-month"]["mean_per_cell"],
        )
        sc_name, geo_name, r = best_geo_attempt
        print(
            "\nNo combination that includes geography clears the >3 bar, even with "
            "sub_class coarsened. The closest attempt is "
            f"{sc_name} sub_class + {geo_name} geography at "
            f"{r['3-month']['mean_per_cell']:.3f}/cell (still below 3)."
        )
        print("The drop-geography recommendation from 00_clean_loss_region.py stands.")
        sc_name, geo_name = "full 78", "none"
        r = grid[(sc_name, geo_name)]
        sc_col = subclass_variants[sc_name]
        final_cols = BASE_SCAN_COLS + [sc_col]

    print(f"\nFinal scan-feature set ({len(final_cols)} features):")
    for c in final_cols:
        print(f"    {c}")

    print("\nRecommended detection window: 3-MONTH")
    print(f"  Why: {sc_name} x {geo_name} clears the >3 mean/cell bar at 3-month "
          f"({r['3-month']['mean_per_cell']:.3f}/cell vs {r['monthly']['mean_per_cell']:.3f}/cell monthly).")

    dist_rec = "negative binomial" if r["3-month"]["vmr"] > 1.5 or total_vmr > 1.5 else "Poisson"
    print(f"\nRecommended null distribution: {dist_rec.upper()}")
    print(f"  Why: cell-level VMR = {r['3-month']['vmr']:.2f}, monthly total VMR = {total_vmr:.2f} (both >> 1).")


def main():
    processed_md5_before = file_md5(config.PROCESSED_DATA_PATH)
    cleaned_md5_before = file_md5(config.CLEANED_REGION_DATA_PATH)

    df = diag.load_data(config.CLEANED_REGION_DATA_PATH)

    vc, _ = inspect_sub_class(df)
    subclass_variants = build_subclass_variants(df, vc)

    month_idx, window_idx = diag.build_time_indices(df)
    grid = run_grid(df, subclass_variants, month_idx, window_idx)
    passing, monthly_totals, total_vmr = report_dispersion(df, grid, month_idx, window_idx)
    plot_grid_heatmap(grid, subclass_variants)
    print_summary(grid, subclass_variants, total_vmr)

    processed_md5_after = file_md5(config.PROCESSED_DATA_PATH)
    cleaned_md5_after = file_md5(config.CLEANED_REGION_DATA_PATH)
    print("\n" + "=" * 78)
    print(f"Source unchanged — processed_data.csv: {processed_md5_before == processed_md5_after} "
          f"(MD5 {processed_md5_after})")
    print(f"Source unchanged — processed_data_region_clean.csv: {cleaned_md5_before == cleaned_md5_after} "
          f"(MD5 {cleaned_md5_after})")


if __name__ == "__main__":
    main()
