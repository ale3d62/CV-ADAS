"""
evaluate_offset.py
------------------
Compares offset test results against ground truth CSVs and prints a
summary table showing MAE, RMSE and MAPE for each offset value.

Requirements (run main_offset_test.py first for each sequence):
  - offset_results/results_waymo10625.csv
  - offset_results/results_waymo10923.csv
  - offset_results/results_waymo11199.csv
  - offset_results/results_waymo13064.csv
  - offset_results/results_waymo13182.csv
  - offset_results/results_kitti15.csv

Run from CV-ADAS/src/:
    python evaluate_offset.py
"""

import os
import csv
import numpy as np

OFFSETS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

GT_DIR     = "../../Resultados"
RESULT_DIR = "offset_results"

SEQUENCES = {
    "waymo10625": "TFM Benchmark Claude - Waymo_10625.csv",
    "waymo10923": "TFM Benchmark Claude - Waymo_10923.csv",
    "waymo11199": "TFM Benchmark Claude - Waymo_11199.csv",
    "waymo13064": "TFM Benchmark Claude - Waymo_13064.csv",
    "waymo13182": "TFM Benchmark Claude - Waymo_13182.csv",
    "kitti15":    "TFM Benchmark Claude - KITTI_15.csv",
}


# ─── CSV parsing helpers ──────────────────────────────────────────────────────

def parse_eu_float(s):
    """Parse European decimal (comma) string to float. Returns None on error."""
    s = s.strip().strip('"')
    if not s or s == "-":
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def read_gt_csv(filepath):
    """
    Read ground truth distances from the benchmark CSV.
    Skips the first 5 header/stats rows. Column 0 = ground truth distance.

    Keeps ONE entry per frame (None when the GT is "-"), so the returned list
    stays aligned frame-by-frame with the estimation CSV read by
    read_offset_results(). compute_metrics() then skips the None (GT "-")
    frames. Compacting the list here (dropping "-" rows) would desynchronise
    GT and estimates and pair each estimate with the wrong frame's GT.
    """
    gt_values = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows[5:]:
        if not row:
            continue
        gt_values.append(parse_eu_float(row[0]))

    return gt_values


def read_offset_results(filepath):
    """
    Read offset test results CSV.
    Returns dict: {offset_float -> list of float|None}
    """
    results = {off: [] for off in OFFSETS}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        return results

    header = lines[0].split(";")
    offset_cols = {}
    for col_idx, col_name in enumerate(header):
        try:
            off_val = float(col_name.replace("offset_", "").replace(",", "."))
            offset_cols[col_idx] = off_val
        except ValueError:
            pass

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(";")
        for col_idx, off_val in offset_cols.items():
            if off_val not in results:
                continue
            val = parse_eu_float(parts[col_idx]) if col_idx < len(parts) else None
            results[off_val].append(val)

    return results


# ─── Metric computation ───────────────────────────────────────────────────────

def compute_metrics(gt_list, est_list):
    """
    Align GT and estimates by index, skipping frames where either is None.
    Returns (MAE, RMSE, MAPE, n_valid).
    """
    errors, pct_errors = [], []

    for gt, est in zip(gt_list, est_list):
        if gt is None or est is None or gt == 0:
            continue
        err = abs(gt - est)
        errors.append(err)
        pct_errors.append(err / gt * 100)

    if not errors:
        return None, None, None, 0

    mae  = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    mape = np.mean(pct_errors)
    return mae, rmse, mape, len(errors)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_metrics = {}  # {sequence -> {offset -> (mae, rmse, mape, n)}}

    for seq_name, gt_filename in SEQUENCES.items():
        gt_path     = os.path.join(GT_DIR, gt_filename)
        result_path = os.path.join(RESULT_DIR, f"results_{seq_name}.csv")

        if not os.path.exists(result_path):
            print(f"  [SKIP] {seq_name}: result file not found ({result_path})")
            continue
        if not os.path.exists(gt_path):
            print(f"  [SKIP] {seq_name}: GT file not found ({gt_path})")
            continue

        gt_vals  = read_gt_csv(gt_path)
        est_dict = read_offset_results(result_path)

        seq_metrics = {}
        for off in OFFSETS:
            est_vals = est_dict[off]
            n = min(len(gt_vals), len(est_vals))
            seq_metrics[off] = compute_metrics(gt_vals[:n], est_vals[:n])

        all_metrics[seq_name] = seq_metrics
        print(f"  [{seq_name}] GT rows: {len(gt_vals)}, "
              f"result rows: {len(est_dict[OFFSETS[0]])}")

    if not all_metrics:
        print("\nNo results found. Run main_offset_test.py first.")
        return

    # ── Per-sequence tables ────────────────────────────────────────────────
    for seq_name, seq_metrics in all_metrics.items():
        print(f"\n{'='*60}")
        print(f"  {seq_name}")
        print(f"{'='*60}")
        print(f"  {'Offset':>8}  {'MAE (m)':>9}  {'RMSE (m)':>9}  {'MAPE (%)':>9}  {'N':>6}")
        print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*6}")
        for off in OFFSETS:
            mae, rmse, mape, n = seq_metrics[off]
            if mae is None:
                print(f"  {off:>8.2f}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {n:>6}")
            else:
                print(f"  {off:>8.2f}  {mae:>9.4f}  {rmse:>9.4f}  {mape:>9.2f}%  {n:>6}")

    # ── Average across all sequences ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  MEDIA SOBRE TODAS LAS SECUENCIAS")
    print(f"{'='*60}")
    print(f"  {'Offset':>8}  {'MAE (m)':>9}  {'RMSE (m)':>9}  {'MAPE (%)':>9}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")

    best_off = None
    best_mae = float("inf")

    for off in OFFSETS:
        maes, rmses, mapes = [], [], []
        for seq_metrics in all_metrics.values():
            mae, rmse, mape, _ = seq_metrics[off]
            if mae is not None:
                maes.append(mae)
                rmses.append(rmse)
                mapes.append(mape)

        if not maes:
            print(f"  {off:>8.2f}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}")
            continue

        avg_mae  = np.mean(maes)
        avg_rmse = np.mean(rmses)
        avg_mape = np.mean(mapes)
        marker = " <- MEJOR" if avg_mae < best_mae else ""
        print(f"  {off:>8.2f}  {avg_mae:>9.4f}  {avg_rmse:>9.4f}  {avg_mape:>9.2f}%{marker}")

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_off = off

    print(f"\n  Offset optimo (menor MAE medio): {best_off:.2f}  ->  MAE = {best_mae:.4f} m")


if __name__ == "__main__":
    main()
