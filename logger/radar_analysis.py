"""
=============================================================================
  RD-03D radar log post-analysis
  -------------------------------------------------------------
  Bachelor's thesis project: RF motion sensor based on ESP32 + RD-03D

  Reads CSV log produced by radar_logger.py and generates ready-to-use
  artifacts for chapter 5 of the thesis:

    1) Console summary table per scenario
    2) Statistics CSV   (one row per scenario, with mean/std/RMSE/bias)
    3) Plots in /reports/<run>/:
         a) summary.png          -- scenario overview bar chart
         b) <scenario>_xy.png    -- top-down XY plot
         c) <scenario>_range.png -- range vs. time
         d) <scenario>_angle.png -- angle vs. time
         e) <scenario>_hist.png  -- error histograms

  Computed metrics (per scenario, per persistent track):
    - n_frames               number of valid frames
    - frame_rate_hz          actual radar update rate
    - false_negative_rate    fraction of frames without expected target(s)
    - false_positive_rate    fraction of frames with extra targets
    - r_mean, r_std          measured range statistics
    - r_bias = mean(r) - exp_r
    - r_rmse = sqrt(mean((r - exp_r)^2))
    - a_mean, a_std          measured angle statistics
    - a_bias, a_rmse         same for angle
    - xy_rmse                Euclidean RMSE in (x, y) plane

  Requirements:
      pip install pandas matplotlib numpy

  Usage:
      python radar_analysis.py logs/radar_20260502_232042.csv
      python radar_analysis.py logs/*.csv          # batch (Linux/macOS)
      python radar_analysis.py logs/r1.csv --no-plots
      python radar_analysis.py logs/r1.csv --output reports/exp1
=============================================================================
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================== Configuration ===============================

# Tableau-friendly palette suitable for thesis printouts
TRACK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Expected radar update rate, used for FN rate calculation when actual rate
# can't be measured (fewer than 2 frames in scenario)
NOMINAL_RATE_HZ = 11.0

# How many degrees we tolerate when matching tracks to expected angle
# (used only for advisory output, not for filtering)
ANGLE_TOLERANCE_DEG = 15.0


# ============================ Loading & cleaning ============================

def load_log(path: Path) -> pd.DataFrame:
    """Load CSV, parse types, return DataFrame indexed by row order."""
    df = pd.read_csv(path)

    # Convert timestamps
    df['wall_time'] = pd.to_datetime(df['wall_time'], errors='coerce')

    # Numeric columns: empty strings -> NaN
    numeric = ['frame_n', 'elapsed_ms', 'exp_n_targets', 'exp_obstacle_mm',
               'exp_range_m', 'exp_angle_deg',
               'r_m', 'x_m', 'y_m', 'angle_deg', 'speed_m_s']
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # String columns: NaN -> empty string for cleaner grouping
    str_cols = ['scenario_id', 'scenario_desc', 'exp_obstacle',
                'target_id', 'event_marker']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    return df


def split_by_scenario(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group rows by scenario_id, dropping the empty-scenario block (pristrelka).
    Returns dict ordered as scenarios appear in the log.
    """
    scenarios: Dict[str, pd.DataFrame] = {}
    for sc_id in df['scenario_id'].unique():
        if not sc_id:
            continue
        scenarios[sc_id] = df[df['scenario_id'] == sc_id].copy()
    return scenarios


# ============================ Per-scenario metrics ==========================

def compute_scenario_stats(sc_df: pd.DataFrame) -> dict:
    """
    Compute one row of summary statistics for a single scenario.
    """
    # Frames in scenario = number of distinct frame_n values
    frames = sc_df['frame_n'].dropna().unique()
    n_frames = len(frames)

    # Frame rate from elapsed time
    elapsed = sc_df['elapsed_ms'].dropna()
    if len(elapsed) >= 2:
        duration_s = (elapsed.max() - elapsed.min()) / 1000.0
        rate = n_frames / duration_s if duration_s > 0 else np.nan
    else:
        duration_s = 0.0
        rate = np.nan

    # Expected values
    exp_r       = sc_df['exp_range_m'].dropna().mean()
    exp_a       = sc_df['exp_angle_deg'].dropna().mean()
    exp_n       = sc_df['exp_n_targets'].dropna().mean()
    obstacle    = sc_df['exp_obstacle'].iloc[0] if len(sc_df) else ''
    obstacle_mm = sc_df['exp_obstacle_mm'].dropna().mean()

    # Filter rows with actual measurements (not empty-frame placeholders)
    meas = sc_df.dropna(subset=['r_m']).copy()

    # Per-track measurements: take mean position across all frames for each track,
    # then identify the "primary" track as the one most frequently present
    if len(meas) > 0:
        track_counts = meas['target_id'].value_counts()
        primary_tid = track_counts.idxmax()
        primary = meas[meas['target_id'] == primary_tid]
    else:
        primary_tid = ''
        primary = meas

    # False-negative & false-positive rate (frame-level)
    # FN: scenario expects ≥1 target, frame has no measurements
    # FP: scenario expects 1 target, frame has ≥2 distinct tracks
    fn_count = 0
    fp_count = 0
    if n_frames > 0 and not np.isnan(exp_n):
        for fn in frames:
            frame_tracks = sc_df[sc_df['frame_n'] == fn]['target_id'].unique()
            frame_tracks = [t for t in frame_tracks if t]
            if len(frame_tracks) < exp_n:
                fn_count += 1
            elif len(frame_tracks) > exp_n:
                fp_count += 1
    fn_rate = fn_count / n_frames if n_frames > 0 else np.nan
    fp_rate = fp_count / n_frames if n_frames > 0 else np.nan

    # Range / angle metrics from primary track only
    def stats_block(values: pd.Series, expected: float) -> dict:
        if len(values) == 0:
            return dict(mean=np.nan, std=np.nan, bias=np.nan, rmse=np.nan)
        v = values.to_numpy()
        m = np.mean(v)
        s = np.std(v, ddof=1) if len(v) > 1 else 0.0
        if np.isnan(expected):
            return dict(mean=m, std=s, bias=np.nan, rmse=np.nan)
        bias = m - expected
        rmse = np.sqrt(np.mean((v - expected) ** 2))
        return dict(mean=m, std=s, bias=bias, rmse=rmse)

    r_st = stats_block(primary['r_m'], exp_r)
    a_st = stats_block(primary['angle_deg'], exp_a)

    # XY RMSE: needs (exp_x, exp_y) reconstructed from polar (exp_r, exp_a)
    # Convention: exp_a is angle from +Y axis, x = r·sin(a), y = r·cos(a)
    if not np.isnan(exp_r) and not np.isnan(exp_a) and len(primary) > 0:
        exp_x = exp_r * np.sin(np.radians(exp_a))
        exp_y = exp_r * np.cos(np.radians(exp_a))
        dx = primary['x_m'].to_numpy() - exp_x
        dy = primary['y_m'].to_numpy() - exp_y
        xy_rmse = float(np.sqrt(np.mean(dx**2 + dy**2)))
    else:
        xy_rmse = np.nan

    return {
        'scenario_id':         sc_df['scenario_id'].iloc[0],
        'description':         sc_df['scenario_desc'].iloc[0],
        'obstacle':            obstacle,
        'obstacle_mm':         obstacle_mm if not np.isnan(obstacle_mm) else '',
        'duration_s':          round(duration_s, 2),
        'n_frames':            n_frames,
        'frame_rate_hz':       round(rate, 2) if not np.isnan(rate) else '',
        'exp_n_targets':       int(exp_n) if not np.isnan(exp_n) else '',
        'exp_range_m':         exp_r,
        'exp_angle_deg':       exp_a,
        'primary_track':       primary_tid,
        'fn_rate':             round(fn_rate, 4),
        'fp_rate':             round(fp_rate, 4),
        'r_mean_m':            round(r_st['mean'], 4) if not np.isnan(r_st['mean']) else '',
        'r_std_m':             round(r_st['std'],  4) if not np.isnan(r_st['std'])  else '',
        'r_bias_m':            round(r_st['bias'], 4) if not np.isnan(r_st['bias']) else '',
        'r_rmse_m':            round(r_st['rmse'], 4) if not np.isnan(r_st['rmse']) else '',
        'a_mean_deg':          round(a_st['mean'], 2) if not np.isnan(a_st['mean']) else '',
        'a_std_deg':           round(a_st['std'],  2) if not np.isnan(a_st['std'])  else '',
        'a_bias_deg':          round(a_st['bias'], 2) if not np.isnan(a_st['bias']) else '',
        'a_rmse_deg':          round(a_st['rmse'], 2) if not np.isnan(a_st['rmse']) else '',
        'xy_rmse_m':           round(xy_rmse, 4) if not np.isnan(xy_rmse) else '',
    }


def build_summary_df(scenarios: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Apply compute_scenario_stats to every scenario, return DataFrame."""
    rows = [compute_scenario_stats(sc_df) for sc_df in scenarios.values()]
    return pd.DataFrame(rows)


# ================================ Plots =====================================

def plot_xy(sc_df: pd.DataFrame, out_path: Path) -> None:
    """Top-down XY plot: each track in its own colour, expected target marked."""
    meas = sc_df.dropna(subset=['x_m', 'y_m'])
    if len(meas) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    # Group by track, plot positions and connecting line
    for i, tid in enumerate(sorted(meas['target_id'].unique())):
        if not tid:
            continue
        tdf = meas[meas['target_id'] == tid].sort_values('elapsed_ms')
        c = TRACK_COLORS[i % len(TRACK_COLORS)]
        ax.plot(tdf['x_m'], tdf['y_m'], '-', color=c, alpha=0.4, linewidth=1.0)
        ax.scatter(tdf['x_m'], tdf['y_m'], s=14, color=c, alpha=0.7, label=tid)

    # Mark expected position
    exp_r = sc_df['exp_range_m'].dropna().mean()
    exp_a = sc_df['exp_angle_deg'].dropna().mean()
    if not np.isnan(exp_r) and not np.isnan(exp_a):
        exp_x = exp_r * np.sin(np.radians(exp_a))
        exp_y = exp_r * np.cos(np.radians(exp_a))
        ax.plot(exp_x, exp_y, marker='*', color='red', markersize=18,
                markeredgecolor='black', label='ожидаемая позиция', zorder=10)

    # Radar origin
    ax.plot(0, 0, marker='^', color='green', markersize=12, zorder=10)
    ax.text(0, -0.08, 'RD-03D', ha='center', va='top', fontsize=9)

    # FOV reference cone ±60°
    for ang in (-60, 0, 60):
        rad = np.radians(ang)
        max_r = max(5.0, meas['r_m'].max() * 1.2 if 'r_m' in meas else 5.0)
        ax.plot([0, max_r * np.sin(rad)], [0, max_r * np.cos(rad)],
                ':', color='gray', alpha=0.3, linewidth=0.7)

    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м (дальность)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f"{sc_df['scenario_id'].iloc[0]}: положения целей")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_range_time(sc_df: pd.DataFrame, out_path: Path) -> None:
    """Range-vs-time plot, with expected value as horizontal line."""
    meas = sc_df.dropna(subset=['r_m'])
    if len(meas) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, tid in enumerate(sorted(meas['target_id'].unique())):
        if not tid:
            continue
        tdf = meas[meas['target_id'] == tid].sort_values('elapsed_ms')
        c = TRACK_COLORS[i % len(TRACK_COLORS)]
        ax.plot(tdf['elapsed_ms'] / 1000, tdf['r_m'], '.-',
                color=c, label=tid, markersize=4, linewidth=0.8)

    exp_r = sc_df['exp_range_m'].dropna().mean()
    if not np.isnan(exp_r):
        ax.axhline(exp_r, color='red', linestyle='--', linewidth=1.2,
                   label=f'ожидаемое = {exp_r:.2f} м')

    ax.set_xlabel('Время от начала сценария, с')
    ax.set_ylabel('Дальность, м')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_title(f"{sc_df['scenario_id'].iloc[0]}: дальность во времени")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_angle_time(sc_df: pd.DataFrame, out_path: Path) -> None:
    meas = sc_df.dropna(subset=['angle_deg'])
    if len(meas) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, tid in enumerate(sorted(meas['target_id'].unique())):
        if not tid:
            continue
        tdf = meas[meas['target_id'] == tid].sort_values('elapsed_ms')
        c = TRACK_COLORS[i % len(TRACK_COLORS)]
        ax.plot(tdf['elapsed_ms'] / 1000, tdf['angle_deg'], '.-',
                color=c, label=tid, markersize=4, linewidth=0.8)

    exp_a = sc_df['exp_angle_deg'].dropna().mean()
    if not np.isnan(exp_a):
        ax.axhline(exp_a, color='red', linestyle='--', linewidth=1.2,
                   label=f'ожидаемый = {exp_a:.1f}°')

    ax.set_xlabel('Время от начала сценария, с')
    ax.set_ylabel('Азимут, градусы')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_title(f"{sc_df['scenario_id'].iloc[0]}: азимут во времени")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_error_hist(sc_df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel histogram: range error and angle error of primary track."""
    meas = sc_df.dropna(subset=['r_m', 'angle_deg'])
    if len(meas) == 0:
        return

    # Pick primary (most frequent) track
    primary_tid = meas['target_id'].value_counts().idxmax()
    primary = meas[meas['target_id'] == primary_tid]

    exp_r = sc_df['exp_range_m'].dropna().mean()
    exp_a = sc_df['exp_angle_deg'].dropna().mean()
    if np.isnan(exp_r) or np.isnan(exp_a):
        return

    err_r = primary['r_m'] - exp_r
    err_a = primary['angle_deg'] - exp_a

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(err_r, bins=20, color='#1f77b4', edgecolor='black', alpha=0.8)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.0)
    axes[0].set_xlabel('Ошибка по дальности, м')
    axes[0].set_ylabel('Кадров')
    axes[0].set_title(f'Гистограмма ошибки дальности '
                      f'(track {primary_tid})')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(err_a, bins=20, color='#ff7f0e', edgecolor='black', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.0)
    axes[1].set_xlabel('Ошибка по азимуту, градусы')
    axes[1].set_ylabel('Кадров')
    axes[1].set_title(f'Гистограмма ошибки азимута '
                      f'(track {primary_tid})')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{sc_df['scenario_id'].iloc[0]}: распределение ошибок")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_summary_bars(summary: pd.DataFrame, out_path: Path) -> None:
    """Bar chart with RMSE per scenario."""
    if summary.empty:
        return
    # Convert empty strings to NaN for plotting
    rdf = summary.copy()
    for col in ('r_rmse_m', 'a_rmse_deg', 'xy_rmse_m'):
        rdf[col] = pd.to_numeric(rdf[col], errors='coerce')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(rdf))
    labels = rdf['scenario_id'].tolist()

    axes[0].bar(x, rdf['r_rmse_m'], color='#1f77b4')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('RMSE дальности, м')
    axes[0].set_title('Точность по дальности')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, rdf['a_rmse_deg'], color='#ff7f0e')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('RMSE азимута, °')
    axes[1].set_title('Точность по азимуту')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(x, rdf['xy_rmse_m'], color='#2ca02c')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    axes[2].set_ylabel('RMSE координат XY, м')
    axes[2].set_title('Суммарная точность XY')
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ================================ Console ===================================

def print_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print('No scenarios found in log.')
        return

    cols = ['scenario_id', 'n_frames', 'frame_rate_hz',
            'fn_rate', 'fp_rate',
            'r_mean_m', 'r_bias_m', 'r_rmse_m',
            'a_mean_deg', 'a_bias_deg', 'a_rmse_deg',
            'xy_rmse_m']
    pretty = summary[cols].copy()
    print()
    print(pretty.to_string(index=False))
    print()


# ================================== Main ====================================

def analyse_one(csv_path: Path, output_dir: Path, make_plots: bool) -> None:
    print(f'=== Analysing {csv_path.name} ===')

    df = load_log(csv_path)
    print(f'Loaded {len(df)} rows, '
          f'{(df["wall_time"].max() - df["wall_time"].min()).total_seconds():.1f} s span')

    scenarios = split_by_scenario(df)
    if not scenarios:
        print('No tagged scenarios found — nothing to analyse.')
        print('(Did you press 1..9 during recording?)')
        return

    print(f'Scenarios in log: {len(scenarios)}')
    for sid, sc_df in scenarios.items():
        print(f'  {sid}: {len(sc_df)} rows')

    summary = build_summary_df(scenarios)
    print_summary(summary)

    # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary CSV
    summary_path = output_dir / 'summary.csv'
    summary.to_csv(summary_path, index=False, encoding='utf-8')
    print(f'  -> {summary_path}')

    # Save plots
    if make_plots:
        plots_dir = output_dir
        plot_summary_bars(summary, plots_dir / 'summary.png')
        print(f'  -> {plots_dir / "summary.png"}')

        for sid, sc_df in scenarios.items():
            safe = sid.replace('/', '_')
            plot_xy        (sc_df, plots_dir / f'{safe}_xy.png')
            plot_range_time(sc_df, plots_dir / f'{safe}_range.png')
            plot_angle_time(sc_df, plots_dir / f'{safe}_angle.png')
            plot_error_hist(sc_df, plots_dir / f'{safe}_hist.png')
            print(f'  -> plots for {sid}')


def main():
    ap = argparse.ArgumentParser(
        description='Analyse RD-03D radar logs (post-experiment).')
    ap.add_argument('csv', nargs='+',
                    help='One or more CSV files produced by radar_logger.py')
    ap.add_argument('--output', type=str, default=None,
                    help='Output directory (default: reports/<csv-stem>/)')
    ap.add_argument('--no-plots', action='store_true',
                    help='Skip plot generation, just print summary table.')
    args = ap.parse_args()

    if getattr(sys, 'frozen', False):
        base_dir = Path(sys.executable).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parent

    for csv_arg in args.csv:
        csv_path = Path(csv_arg)
        if not csv_path.exists():
            print(f'[ERROR] Not found: {csv_path}', file=sys.stderr)
            continue

        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = base_dir / 'reports' / csv_path.stem

        try:
            analyse_one(csv_path, output_dir, make_plots=not args.no_plots)
        except Exception as e:
            print(f'[ERROR] Failed on {csv_path.name}: {e}', file=sys.stderr)
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
