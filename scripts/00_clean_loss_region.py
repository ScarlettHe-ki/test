"""Clean loss_region's data-quality tail, coarsen it to US Census regions and
divisions, and check whether coarsening clears the >3 mean-claims/cell bar
that the raw 72-value loss_region failed at (see 01_sparsity_dispersion_check.py).

Read-only on config.PROCESSED_DATA_PATH: writes the cleaned + coarsened table
to a NEW file (config.CLEANED_REGION_DATA_PATH), never overwrites the source.
Reuses cell-counting and plotting helpers from 01_sparsity_dispersion_check.py
rather than reimplementing them.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# ---------------------------------------------------------------------------
# STEP 2 reference data — built from inspecting the actual loss_region values
# (see STEP 1 output), not assumed in advance.
# ---------------------------------------------------------------------------
STATE_NAME_TO_CODE = {
    "Florida": "FL",
    "California": "CA",
    "South Carolina": "SC",
    "Alabama": "AL",
    "New Jersey": "NJ",
    "Rhode Island": "RI",
}
MALFORMED_TO_CODE = {
    "TX BGHC1": "TX",
}
CITY_TO_STATE = {
    "Orlando": "FL",
    "Scotts Valley": "CA",
    "McKinney": "TX",
    "St. Louis": "MO",
    "Brooklyn": "NY",
}
# Cities with more than one plausible US match — excluded rather than guessed.
AMBIGUOUS_CITIES = {
    "New Milford": ["CT", "NJ", "PA"],
}
MULTI_REGION_ROW_SHARE_EXCLUDE_THRESHOLD = 0.005  # 0.5%

# ---------------------------------------------------------------------------
# STEP 3 reference data — standard US Census Bureau state -> division -> region.
# ---------------------------------------------------------------------------
STATE_TO_DIVISION = {
    "CT": "New England", "ME": "New England", "MA": "New England",
    "NH": "New England", "RI": "New England", "VT": "New England",
    "NJ": "Middle Atlantic", "NY": "Middle Atlantic", "PA": "Middle Atlantic",
    "IL": "East North Central", "IN": "East North Central", "MI": "East North Central",
    "OH": "East North Central", "WI": "East North Central",
    "IA": "West North Central", "KS": "West North Central", "MN": "West North Central",
    "MO": "West North Central", "NE": "West North Central", "ND": "West North Central",
    "SD": "West North Central",
    "DE": "South Atlantic", "DC": "South Atlantic", "FL": "South Atlantic",
    "GA": "South Atlantic", "MD": "South Atlantic", "NC": "South Atlantic",
    "SC": "South Atlantic", "VA": "South Atlantic", "WV": "South Atlantic",
    "AL": "East South Central", "KY": "East South Central", "MS": "East South Central",
    "TN": "East South Central",
    "AR": "West South Central", "LA": "West South Central", "OK": "West South Central",
    "TX": "West South Central",
    "AZ": "Mountain", "CO": "Mountain", "ID": "Mountain", "MT": "Mountain",
    "NV": "Mountain", "NM": "Mountain", "UT": "Mountain", "WY": "Mountain",
    "AK": "Pacific", "CA": "Pacific", "HI": "Pacific", "OR": "Pacific", "WA": "Pacific",
}
DIVISION_TO_REGION = {
    "New England": "Northeast", "Middle Atlantic": "Northeast",
    "East North Central": "Midwest", "West North Central": "Midwest",
    "South Atlantic": "South", "East South Central": "South", "West South Central": "South",
    "Mountain": "West", "Pacific": "West",
}

# Recorded from the loss_region run in 01_sparsity_dispersion_check.py, for
# the granularity comparison in Step 5 (that run used the raw 72-value field).
PRIOR_REGION_RUN_REFERENCE = {
    "label": "loss_region (72 values, uncleaned)",
    "cardinality": 72,
    "monthly": {"frac_1": 0.750, "mean_per_cell": 1.950, "vmr": 8.820},
    "3-month": {"frac_1": 0.719, "mean_per_cell": 2.052, "vmr": 10.391},
}

BASE_SCAN_COLS = [
    "peril_type", "syndicate", "group_class", "sub_class",
    "placing_basis_group", "leader_status", "new_renewal", "risk_code",
]


def classify_value(value):
    if pd.isna(value):
        return "missing"
    if len(value) == 2 and value.isupper() and value.isalpha():
        return "clean"
    if "|" in value:
        return "multi"
    if value in STATE_NAME_TO_CODE:
        return "full_name"
    if value in MALFORMED_TO_CODE:
        return "malformed"
    if value in CITY_TO_STATE:
        return "city_confident"
    if value in AMBIGUOUS_CITIES:
        return "city_ambiguous"
    return "unclassified"


def inspect_loss_region(df):
    print("=" * 78)
    print("STEP 1 — INSPECT loss_region (72 distinct values)")
    print("=" * 78)

    vc = df["loss_region"].value_counts()
    categories = {}
    for value, count in vc.items():
        categories.setdefault(classify_value(value), []).append((value, count))

    labels = {
        "clean": "Clean 2-letter state codes",
        "full_name": "Full state names",
        "malformed": "Malformed codes",
        "city_confident": "Stray city names (confidently mappable)",
        "city_ambiguous": "Stray city names (ambiguous, no confident map)",
        "multi": "Multi-region combos (pipe-delimited)",
        "unclassified": "UNCLASSIFIED (needs attention)",
    }
    n_rows = len(df)
    for key, label in labels.items():
        items = categories.get(key, [])
        rows = sum(c for _, c in items)
        print(f"\n{label}: {len(items)} distinct values, {rows} rows ({rows / n_rows:.2%})")
        for value, count in items:
            print(f"    {value!r:40s} {count:5d}")

    if categories.get("unclassified"):
        raise ValueError(
            f"Unclassified loss_region values found: {categories['unclassified']} "
            "— extend the lookup tables before proceeding."
        )
    return categories


def resolve_multi_region_mode(categories, n_rows):
    multi_rows = sum(c for _, c in categories.get("multi", []))
    share = multi_rows / n_rows
    mode = "exclude" if share < MULTI_REGION_ROW_SHARE_EXCLUDE_THRESHOLD else "first"
    print(
        f"\nMulti-region combos cover {multi_rows} rows ({share:.4%} of data), "
        f"below the {MULTI_REGION_ROW_SHARE_EXCLUDE_THRESHOLD:.1%} threshold "
        f"-> default mode = '{mode}'."
    )
    print(
        "  Both resolution options are implemented (assign_first_listed / "
        f"exclude); '{mode}' fired by default. Override MULTI_REGION_MODE_OVERRIDE "
        "in this script to force the other."
    )
    return mode


MULTI_REGION_MODE_OVERRIDE = None  # set to "first" or "exclude" to force a mode


def clean_state_codes(df, categories):
    print("\n" + "=" * 78)
    print("STEP 2 — CLEAN to canonical 2-letter state codes")
    print("=" * 78)

    print("\nFull-name lookup used:")
    for name, code in STATE_NAME_TO_CODE.items():
        print(f"    {name!r:20s} -> {code}")
    print("\nMalformed-code lookup used:")
    for raw, code in MALFORMED_TO_CODE.items():
        print(f"    {raw!r:20s} -> {code}")
    print("\nCity lookup used (confidently mapped):")
    for city, code in CITY_TO_STATE.items():
        print(f"    {city!r:20s} -> {code}")
    if AMBIGUOUS_CITIES:
        print("\nCities NOT mapped (ambiguous, excluded instead of guessed):")
        for city, candidates in AMBIGUOUS_CITIES.items():
            print(f"    {city!r:20s} candidates: {candidates}")

    n_rows = len(df)
    mode = MULTI_REGION_MODE_OVERRIDE or resolve_multi_region_mode(categories, n_rows)

    def resolve(value):
        cat = classify_value(value)
        if cat == "clean":
            return value
        if cat == "full_name":
            return STATE_NAME_TO_CODE[value]
        if cat == "malformed":
            return MALFORMED_TO_CODE[value]
        if cat == "city_confident":
            return CITY_TO_STATE[value]
        if cat == "city_ambiguous":
            return None
        if cat == "multi":
            return value.split("|")[0].strip() if mode == "first" else None
        return None

    cleaned = df.copy()
    cleaned["loss_region"] = cleaned["loss_region"].map(resolve)

    excluded_mask = cleaned["loss_region"].isna()
    excluded_by_reason = {}
    for value, count in df.loc[excluded_mask, "loss_region"].value_counts().items():
        excluded_by_reason[value] = count
    print(f"\nRows excluded ({excluded_mask.sum()} total):")
    for value, count in excluded_by_reason.items():
        print(f"    {value!r:40s} {count:5d}")

    cleaned = cleaned.loc[~excluded_mask].reset_index(drop=True)

    residual = cleaned["loss_region"][~cleaned["loss_region"].str.match(r"^[A-Z]{2}$")]
    if len(residual):
        raise AssertionError(f"Residual non-conforming loss_region values: {residual.unique()}")
    print(
        f"\nAssertion passed: all {cleaned['loss_region'].nunique()} remaining "
        "loss_region values are clean 2-letter codes."
    )
    print(f"Row count: {n_rows} -> {len(cleaned)} ({n_rows - len(cleaned)} dropped)")
    return cleaned


def coarsen_to_census(df):
    print("\n" + "=" * 78)
    print("STEP 3 — COARSEN to US Census regions and divisions")
    print("=" * 78)

    unmapped = set(df["loss_region"].unique()) - set(STATE_TO_DIVISION)
    if unmapped:
        raise ValueError(f"States with no division mapping: {unmapped}")

    df = df.copy()
    df["loss_census_division"] = df["loss_region"].map(STATE_TO_DIVISION)
    df["loss_census_region"] = df["loss_census_division"].map(DIVISION_TO_REGION)

    print("\nstate -> division map:")
    for division in sorted(set(STATE_TO_DIVISION.values())):
        states = sorted(s for s, d in STATE_TO_DIVISION.items() if d == division)
        print(f"    {division:20s} {states}")
    print("\ndivision -> region map:")
    for region in sorted(set(DIVISION_TO_REGION.values())):
        divisions = sorted(d for d, r in DIVISION_TO_REGION.items() if r == region)
        print(f"    {region:10s} {divisions}")

    print(f"\nloss_census_division row counts ({df['loss_census_division'].nunique()} buckets):")
    print(df["loss_census_division"].value_counts().to_string())
    print(f"\nloss_census_region row counts ({df['loss_census_region'].nunique()} buckets):")
    print(df["loss_census_region"].value_counts().to_string())

    return df


def write_output(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nWrote cleaned + coarsened data: {path} ({len(df)} rows, {df.shape[1]} columns)")


def evaluate_geography_scenario(df, label, geo_col, month_idx, window_idx):
    cols = BASE_SCAN_COLS + ([geo_col] if geo_col else [])
    monthly_counts = diag.cell_counts(df, cols, month_idx)
    window_counts = diag.cell_counts(df, cols, window_idx)
    monthly_dist = diag.cell_distribution(monthly_counts)
    window_dist = diag.cell_distribution(window_counts)
    monthly_vmr = diag._vmr(monthly_counts)
    window_vmr = diag._vmr(window_counts)
    cardinality = df[geo_col].nunique() if geo_col else 1

    usable = window_dist["mean_per_cell"] > diag.MEAN_PER_CELL_USABLE_THRESHOLD
    print(f"\n--- {label} ({len(cols)} features, geo cardinality={cardinality}) ---")
    print(f"  {'':18s}{'monthly':>14s}{'3-month':>14s}")
    print(f"  {'populated cells':18s}{monthly_dist['n_populated_cells']:14,d}{window_dist['n_populated_cells']:14,d}")
    print(f"  {'frac count==1':18s}{monthly_dist['frac_1']:14.3f}{window_dist['frac_1']:14.3f}")
    print(f"  {'mean/cell':18s}{monthly_dist['mean_per_cell']:14.3f}{window_dist['mean_per_cell']:14.3f}")
    print(f"  {'VMR':18s}{monthly_vmr:14.3f}{window_vmr:14.3f}")
    print(f"  3-month >3 usable? {'YES' if usable else 'NO'} ({window_dist['mean_per_cell']:.3f}/cell)")

    return {
        "label": label,
        "cardinality": cardinality,
        "monthly": {**monthly_dist, "vmr": monthly_vmr},
        "3-month": {**window_dist, "vmr": window_vmr},
        "usable_3month": usable,
    }


def report_diagnostics_comparison(df):
    print("\n" + "=" * 78)
    print("STEP 4 — RE-RUN SPARSITY/DISPERSION AT BOTH GRANULARITIES")
    print("=" * 78)

    month_idx, window_idx = diag.build_time_indices(df)

    results = [
        evaluate_geography_scenario(df, "9 divisions (loss_census_division)", "loss_census_division", month_idx, window_idx),
        evaluate_geography_scenario(df, "4 regions (loss_census_region)", "loss_census_region", month_idx, window_idx),
        evaluate_geography_scenario(df, "no geography (baseline)", None, month_idx, window_idx),
    ]

    monthly_totals = df.groupby(df["claim_date"].dt.to_period("M")).size().sort_index()
    total_vmr = monthly_totals.var() / monthly_totals.mean()
    z = (monthly_totals - monthly_totals.mean()) / monthly_totals.std()
    outliers = monthly_totals[z.abs() > 2]
    print(f"\nMonthly TOTAL claim count series (cleaned data): mean={monthly_totals.mean():.1f}, "
          f"var={monthly_totals.var():.1f}, variance/mean={total_vmr:.2f}")
    if len(outliers):
        print("Outlier months (|z| > 2):")
        for period, count in outliers.items():
            print(f"  {period}: {count} claims (z={z[period]:.2f})")

    diag.plot_monthly_totals(monthly_totals, z, filename="monthly_total_claims_region_coarse.png")

    sparsity_results_for_plot = {r["label"]: {"monthly": r["monthly"], "3-month": r["3-month"]} for r in results}
    diag.plot_cell_distributions(sparsity_results_for_plot, filename="cell_count_distribution_region_coarse.png")

    return results, total_vmr, outliers


def print_summary(df, results, total_vmr, outliers):
    print("\n" + "=" * 78)
    print("STEP 5 — SUMMARY / DECISION")
    print("=" * 78)

    rows = [
        (
            PRIOR_REGION_RUN_REFERENCE["label"],
            PRIOR_REGION_RUN_REFERENCE["cardinality"],
            PRIOR_REGION_RUN_REFERENCE["3-month"]["frac_1"],
            PRIOR_REGION_RUN_REFERENCE["3-month"]["mean_per_cell"],
            PRIOR_REGION_RUN_REFERENCE["3-month"]["mean_per_cell"] > diag.MEAN_PER_CELL_USABLE_THRESHOLD,
        )
    ]
    for r in results:
        rows.append((r["label"], r["cardinality"], r["3-month"]["frac_1"], r["3-month"]["mean_per_cell"], r["usable_3month"]))

    print(f"\n{'geography':38s}{'cardinality':>12s}{'singleton %':>13s}{'mean/cell':>11s}{'>3 verdict':>12s}")
    for label, card, frac1, mean, usable in rows:
        verdict = "PASS" if usable else "fail"
        print(f"  {label:36s}{card:12d}{frac1:13.1%}{mean:11.3f}{verdict:>12s}")

    division_result = next(r for r in results if "division" in r["label"])
    region_result = next(r for r in results if "4 regions" in r["label"])
    baseline_result = next(r for r in results if "baseline" in r["label"])

    if division_result["usable_3month"]:
        chosen = division_result
        geo_col = "loss_census_division"
    elif region_result["usable_3month"]:
        chosen = region_result
        geo_col = "loss_census_region"
    else:
        chosen = baseline_result
        geo_col = None

    print(f"\nRecommendation: {chosen['label']}")
    if geo_col:
        print(
            f"  Coarsest granularity that clears the >3 bar: {geo_col} "
            f"({chosen['cardinality']} categories, 3-month mean/cell={chosen['3-month']['mean_per_cell']:.3f})."
        )
    else:
        print(
            "  Even 4 Census regions fails the >3 bar "
            f"({region_result['3-month']['mean_per_cell']:.3f}/cell) — recommend dropping "
            "geography from the scan-feature set entirely."
        )

    final_cols = BASE_SCAN_COLS + ([geo_col] if geo_col else [])
    print(f"\nFinal recommended scan-feature set ({len(final_cols)} features):")
    for c in sorted(final_cols, key=lambda c: df[c].nunique()):
        print(f"    {c:24s} {df[c].nunique():5d} distinct values")

    max_vmr = max(max(r["monthly"]["vmr"], r["3-month"]["vmr"]) for r in results)
    dist_rec = "negative binomial" if max_vmr > 1.5 or total_vmr > 1.5 else "Poisson"
    print(f"\nRecommended detection window: 3-MONTH")
    print(f"  Why: {chosen['label']} clears the >3 mean/cell usability bar at 3-month "
          f"({chosen['3-month']['mean_per_cell']:.3f}/cell vs {chosen['monthly']['mean_per_cell']:.3f}/cell monthly).")
    print(f"\nRecommended null distribution: {dist_rec.upper()}")
    print(f"  Why: cell-level VMR up to {max_vmr:.2f}, monthly total VMR = {total_vmr:.2f} (both >> 1).")
    if len(outliers):
        print(f"  Candidate catastrophe/event months: {', '.join(str(p) for p in outliers.index)}")


def main():
    source_md5_before = file_md5(config.PROCESSED_DATA_PATH)

    df = diag.load_data(config.PROCESSED_DATA_PATH)

    categories = inspect_loss_region(df)
    cleaned = clean_state_codes(df, categories)
    coarsened = coarsen_to_census(cleaned)
    write_output(coarsened, config.CLEANED_REGION_DATA_PATH)

    results, total_vmr, outliers = report_diagnostics_comparison(coarsened)
    print_summary(coarsened, results, total_vmr, outliers)

    source_md5_after = file_md5(config.PROCESSED_DATA_PATH)
    print("\n" + "=" * 78)
    print(f"Source file unchanged: {source_md5_before == source_md5_after} "
          f"(MD5 {source_md5_after}, {config.PROCESSED_DATA_PATH})")


if __name__ == "__main__":
    main()
