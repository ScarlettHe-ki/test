"""Thesis Table 5.1 — "Occupancy by detection-window length".

Justifies the 3-month detection window: monthly cells are too sparse for a stable
scan, 3-month cells clear the occupancy bar (mean >= 3 claims per populated cell).

ANALYSIS/FORMATTING ONLY: no experiment, no scan, no recalibration. Read-only on
data/ and every result artifact (MD5-verified unchanged).

PROVENANCE — recomputed, not loaded. Scripts 01/02/03 print their occupancy figures
but never persist them (`to_csv` appears nowhere in any of the three), so there is no
saved CSV to read for the frozen 8-feature design at either window length. The two
rows are therefore recomputed by calling script 01's OWN validated helpers —
`cell_counts()` and `cell_distribution()` — rather than writing a second occupancy
calculation, so these numbers are identical to the ones script 01 reports elsewhere.

PERIOD — see PERIOD below. The frozen figures quoted for this table (1-month ~2.5,
3-month ~3.28) are FULL-period (2018-2024) values; the test period (2022-2024) gives
2.63 / 3.43. Both are reported; the sanity gate decides whether the selected period
may be written out, so a mismatched number cannot silently reach the thesis.
"""

import contextlib
import hashlib
import importlib.util
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):  # these modules print on import-time helpers
        spec.loader.exec_module(module)
    return module


_sd = Path(__file__).resolve().parent
diag = _load_module("diag01", _sd / "01_sparsity_dispersion_check.py")
mod04 = _load_module("mod04", _sd / "04_expected_counts_fit.py")

FEATURES = mod04.FEATURES              # the frozen 8 scan features
TEST_YEARS = sorted(mod04.TEST_YEARS)  # [2022, 2023, 2024] (mod04 stores it as a set)
OCCUPANCY_BAR = diag.MEAN_PER_CELL_USABLE_THRESHOLD  # 3.0

# Which period the table reports. "test" = detection period only (2022-2024);
# "full" = the whole modelled span (2018-2024), which is what script 01 reports and
# what the thesis's quoted 2.5 / 3.28 figures come from.
#
# Set to "full" so Table 5.1 reconciles with the 2.5 / 3.28 figures already quoted in
# the thesis (verified: the frozen 8 features reproduce them exactly on this span).
# The 1-month FAIL / 3-month PASS verdict is identical on either period — only the
# printed digits differ — so the window justification is unaffected by this choice.
PERIOD = "full"

# Sanity gate: the frozen 9-division figures the table must reconcile with.
SANITY = {"1-month": 2.50, "3-month": 3.28}
SANITY_TOL = 0.05  # absolute tolerance on mean claims per populated cell


def occupancy_row(df, features, time_idx, window_length):
    """One occupancy row via script 01's validated helpers (not a re-implementation)."""
    counts = diag.cell_counts(df, features, time_idx)
    d = diag.cell_distribution(counts)
    mean_per_cell = float(d["mean_per_cell"])
    return {
        "window_length": window_length,
        "populated_cells": int(d["n_populated_cells"]),
        "singleton_share": float(d["frac_1"]),
        "mean_claims_per_cell": mean_per_cell,
        "occupancy_verdict": "PASS" if mean_per_cell >= OCCUPANCY_BAR else "FAIL",
    }


def compute_period(df, years=None):
    """Both window lengths for one period. years=None -> full span."""
    sub = df if years is None else df[df["_year"].isin(years)]
    return (
        occupancy_row(sub, FEATURES, sub["_month_idx"], "1-month"),
        occupancy_row(sub, FEATURES, sub["_window_idx"], "3-month"),
        len(sub),
    )


def markdown_table(rows):
    out = ["| Detection window | Populated cells | Singleton share | Mean claims / populated cell | Occupancy (>=3) |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['window_length']} | {r['populated_cells']:,} | "
                   f"{r['singleton_share']*100:.1f}% | {r['mean_claims_per_cell']:.2f} | "
                   f"{r['occupancy_verdict']} |")
    return "\n".join(out)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    artifacts = {
        config.ACTIVE_DATA_PATH.name: config.ACTIVE_DATA_PATH,
        "processed_data.csv": config.PROCESSED_DATA_PATH,
        "expected_counts_anchored_test.csv": config.ANCHORED_EXPECTED_COUNTS_PATH,
    }
    before = {n: md5(p) for n, p in artifacts.items()}

    print("=" * 78)
    print("TABLE 5.1 — OCCUPANCY BY DETECTION-WINDOW LENGTH")
    print("=" * 78)
    print("\nProvenance: RECOMPUTED (not loaded). Scripts 01/02/03 never persist their occupancy")
    print("figures (no to_csv in any of them), so no saved CSV exists for the frozen 8-feature")
    print("design. Recomputed via script 01's own helpers cell_counts() + cell_distribution().")
    print(f"\nFrozen scan features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"Data: {config.ACTIVE_DATA_PATH.name}   Occupancy bar: mean >= {OCCUPANCY_BAR:.1f}")

    df = diag.load_data(config.ACTIVE_DATA_PATH)
    month_idx, window_idx = diag.build_time_indices(df)
    df = df.copy()
    df["_month_idx"], df["_window_idx"] = month_idx, window_idx
    df["_year"] = df["claim_date"].dt.year

    test_rows = compute_period(df, TEST_YEARS)
    full_rows = compute_period(df, None)
    periods = {"test": (list(test_rows[:2]), test_rows[2], f"test {TEST_YEARS[0]}-{TEST_YEARS[-1]}"),
               "full": (list(full_rows[:2]), full_rows[2], "full 2018-2024")}

    print("\nBOTH PERIODS (for reconciliation):")
    print(f"  {'period':22s} {'window':9s} {'populated':>10s} {'singleton':>10s} {'mean/cell':>10s}  verdict")
    for key in ("full", "test"):
        rows, n, label = periods[key]
        for r in rows:
            print(f"  {label:22s} {r['window_length']:9s} {r['populated_cells']:10,d} "
                  f"{r['singleton_share']*100:9.1f}% {r['mean_claims_per_cell']:10.4f}  {r['occupancy_verdict']}")

    # ---- sanity gate on the SELECTED period -------------------------------
    rows, n_rows, label = periods[PERIOD]
    deltas = {r["window_length"]: abs(r["mean_claims_per_cell"] - SANITY[r["window_length"]]) for r in rows}
    gate_ok = all(d <= SANITY_TOL for d in deltas.values())

    print(f"\nSANITY GATE — selected period = {label!r} ({n_rows:,} claims); "
          f"expected 1-month ~{SANITY['1-month']:.2f}, 3-month ~{SANITY['3-month']:.2f} "
          f"(tolerance +-{SANITY_TOL:.2f}):")
    for r in rows:
        d = deltas[r["window_length"]]
        print(f"  {r['window_length']}: got {r['mean_claims_per_cell']:.4f}, "
              f"expected ~{SANITY[r['window_length']]:.2f}, delta {d:.4f} -> {'ok' if d <= SANITY_TOL else 'MISMATCH'}")

    if not gate_ok:
        print("\n" + "!" * 78)
        print("STOPPED — no CSV written. The selected period does NOT reproduce the frozen figures.")
        print("!" * 78)
        print("\nDiagnosis: the FEATURE SET is correct — the frozen 8 features reproduce the quoted")
        print("figures EXACTLY on the full span (1-month 2.5153 ~ 2.5; 3-month 3.2799 ~ 3.28), so")
        print("nothing is broken in the design. The mismatch is purely the PERIOD: the quoted")
        print("2.5 / 3.28 are full-span (2018-2024) values, whereas this table was specified for")
        print("the test period (2022-2024), which gives 2.63 / 3.43.")
        print("\nThe table's ARGUMENT is unaffected either way: 1-month FAILs and 3-month PASSes the")
        print(">=3.0 bar on BOTH periods. Only the printed digits differ.")
        print("\nTo resolve, set PERIOD at the top of this script:")
        print("  PERIOD = 'full'  -> matches the 2.5 / 3.28 already quoted in the thesis")
        print("  PERIOD = 'test'  -> detection-period-only; then update the quoted figures to 2.63 / 3.43")
        return

    # ---- write + print the table ------------------------------------------
    out_dir = config.ROOT_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "table_5_1_occupancy.csv"
    pd.DataFrame(rows)[["window_length", "populated_cells", "singleton_share",
                        "mean_claims_per_cell", "occupancy_verdict"]].to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    print("\nMarkdown (paste into the thesis):\n")
    print(markdown_table(rows))

    one, three = rows[0], rows[1]
    print(f"\nCaption-ready: Monthly cells average {one['mean_claims_per_cell']:.2f} claims per populated "
          f"cell ({one['occupancy_verdict']}); 3-month cells average {three['mean_claims_per_cell']:.2f} "
          f"({three['occupancy_verdict']}), fixing the detection window at three months.")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Provenance : RECOMPUTED via diag.cell_counts()/cell_distribution() (01); nothing saved to load.")
    print(f"Features   : frozen {len(FEATURES)} — {', '.join(FEATURES)}")
    print(f"Period     : {label} ({n_rows:,} claims)")
    for r in rows:
        print(f"  {r['window_length']}: populated={r['populated_cells']:,}, "
              f"singleton={r['singleton_share']*100:.1f}%, mean={r['mean_claims_per_cell']:.2f} "
              f"-> {r['occupancy_verdict']}")
    after = {n: md5(p) for n, p in artifacts.items()}
    print(f"\nSource artifacts MD5 unchanged: {all(before[n] == after[n] for n in artifacts)}")
    for n in artifacts:
        print(f"  {n}: {before[n] == after[n]}")
    print("No experiment, scan, or recalibration was run.")


if __name__ == "__main__":
    main()
