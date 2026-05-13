"""
Baseline Models — Heuritech F1 Fashion Dataset

Three baselines required by any credible forecasting paper, evaluated on
the 40-week test split (2019-04-01 → 2019-12-30):

  1. Naive (random walk)    – repeat the last observed value
  2. Seasonal naive (SNaive)– repeat values from exactly 52 weeks ago
  3. ARIMA(1,1,1)           – fit per series via pmdarima, predict forward

Naive and SNaive are O(1) per series and run across all 10 000 series.
ARIMA is O(T²) per series; it is applied to a stratified random sample
of 500 series balanced across HERMES trend labels:
  • increasing  : 167 series
  • flat        : 167 series
  • declining   : 166 series

A fair head-to-head comparison is built on the 500-series ARIMA subsample.
The full 10 000-series naive / SNaive results are also saved for reference.

Results saved to results/baselines.csv.
"""

import json
import os
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths & constants ──────────────────────────────────────────────────────
PARQUET     = "data/f1_fashion/f1_cleaned.parquet"
SPLITS_JSON = "data/f1_fashion/splits.json"
RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "baselines.csv")

SEED        = 42
N_SAMPLE    = 500          # total ARIMA series
FREQ        = 52           # weekly seasonality
THRESHOLD   = 0.05         # HERMES YoY threshold (David et al. 2022)


# ── Data loading ───────────────────────────────────────────────────────────

def _load_data(parquet_path: str = PARQUET, splits_path: str = SPLITS_JSON):
    """
    Load parquet and split indices; return wide DataFrames and metadata.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame  (timesteps × 10 000)
    meta                      : dict with index boundaries and dates
    """
    with open(splits_path) as fh:
        meta = json.load(fh)

    idx = meta["indices"]
    df_long = pd.read_parquet(parquet_path)
    main = df_long[df_long["signal"] == "main"]
    wide = main.pivot(index="date", columns="series_id", values="value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    train_df = wide.iloc[idx["train_start"]: idx["train_end"]]
    val_df   = wide.iloc[idx["val_start"]  : idx["val_end"]]
    test_df  = wide.iloc[idx["test_start"] : idx["test_end"]]

    return train_df, val_df, test_df, meta


def _series_meta(series_id: str) -> Dict[str, str]:
    """Parse market / gender / category from series_id."""
    market, gender, category, n = series_id.split("_", 3)
    return {"market": market, "gender": gender, "category": category, "n": int(n)}


# ── HERMES trend labels ────────────────────────────────────────────────────

def _compute_trend_labels(
    history_df: pd.DataFrame,
    threshold: float = THRESHOLD,
    freq: int = FREQ,
) -> pd.Series:
    """
    Year-on-year classification from David et al. (2022) metrics.py.

    yoy   = (mean(last_freq) - mean(prev_freq)) / mean(prev_freq)
    label =  1  if yoy >  threshold   → increasing
           = -1  if yoy < -threshold  → declining
           =  0  otherwise            → flat

    `history_df` is the combined train+val window (221 timesteps).
    """
    last = history_df.iloc[-freq:]
    prev = history_df.iloc[-2 * freq: -freq]

    mean_last = last.mean()
    mean_prev = prev.mean().replace(0, np.nan)

    yoy = (mean_last - mean_prev) / mean_prev
    yoy = yoy.fillna(0.0)

    labels = (yoy > threshold).astype(int) - (yoy < -threshold).astype(int)
    return labels


# ── Stratified sample ──────────────────────────────────────────────────────

def _stratified_sample(
    labels: pd.Series,
    n_total: int = N_SAMPLE,
    seed: int = SEED,
) -> List[str]:
    """
    Return `n_total` series_ids balanced across HERMES trend classes
    (increasing=1, flat=0, declining=-1).

    Allocation: floor(n_total / 3) per class; remainder assigned to the
    first classes in sorted order to reach exactly n_total.
    """
    rng   = np.random.default_rng(seed)
    classes = sorted(labels.unique())           # [-1, 0, 1]
    n_classes = len(classes)
    base  = n_total // n_classes               # 166
    extra = n_total  % n_classes               # 2

    sampled = []
    for i, cls in enumerate(classes):
        pool = labels[labels == cls].index.tolist()
        k    = base + (1 if i < extra else 0)
        k    = min(k, len(pool))               # guard against small classes
        chosen = rng.choice(pool, size=k, replace=False).tolist()
        sampled.extend(chosen)

    return sampled


# ── Baseline predictors ────────────────────────────────────────────────────

def predict_naive(history: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Naive / random-walk baseline.
    Forecast = last observed value, held constant for all horizons.
    Optimal under a random-walk (unit-root, no drift) data-generating
    process; a strong benchmark when trends are unpredictable.
    """
    return np.full(n_steps, history[-1])


def predict_seasonal_naive(
    history: np.ndarray,
    n_steps: int,
    freq: int = FREQ,
) -> np.ndarray:
    """
    Seasonal naive baseline (SNaive).
    Forecast_t = value observed freq steps ago in the history window.
    Tiles the last `freq` history values as needed; handles n_steps < freq.
    Benchmark of choice when series exhibit strong annual seasonality.
    """
    season = history[-freq:]
    reps   = int(np.ceil(n_steps / freq))
    return np.tile(season, reps)[:n_steps]


def predict_arima(history: np.ndarray, n_steps: int) -> np.ndarray:
    """
    ARIMA(1,1,1) baseline fitted with pmdarima.
    Order is fixed (p=1, d=1, q=1); no automatic model selection so that
    the same inductive bias is applied uniformly across all sampled series.
    Falls back to the naive forecast if fitting fails (e.g. constant series).
    """
    from pmdarima import ARIMA as PmARIMA

    try:
        model = PmARIMA(order=(1, 1, 1), suppress_warnings=True)
        model.fit(history)
        return model.predict(n_periods=n_steps)
    except Exception:
        return predict_naive(history, n_steps)


# ── Per-series evaluation ──────────────────────────────────────────────────

def _evaluate_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    series_id: str,
    model: str,
    trend_label: int,
) -> Dict:
    """Compute RMSE and R² for one series / model pair."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    m    = _series_meta(series_id)
    label_name = {1: "increasing", 0: "flat", -1: "declining"}[trend_label]
    return {
        "series_id":   series_id,
        "market":      m["market"],
        "gender":      m["gender"],
        "category":    m["category"],
        "trend_label": label_name,
        "model":       model,
        "rmse":        rmse,
        "r2":          r2,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_baselines(
    parquet_path: str = PARQUET,
    splits_path:  str = SPLITS_JSON,
    results_path: str = RESULTS_CSV,
    n_sample:     int = N_SAMPLE,
    seed:         int = SEED,
) -> pd.DataFrame:
    """
    Run all three baselines and save results.

    Returns the combined results DataFrame (also saved as CSV).
    """
    # ── Load data ────────────────────────────────────────────────────────
    print("=" * 65)
    print("BASELINE EVALUATION — F1 Fashion Dataset")
    print("=" * 65)

    print("\nLoading data …")
    train_df, val_df, test_df, meta = _load_data(parquet_path, splits_path)
    history_df = pd.concat([train_df, val_df])   # 221 timesteps
    n_test     = len(test_df)                     # 40 timesteps
    n_total    = len(history_df.columns)          # 10 000 series

    print(f"  History  : {len(history_df)} weeks (train + val)")
    print(f"  Test     : {n_test} weeks")
    print(f"  Series   : {n_total:,}")

    # ── Trend labels & stratified sample ─────────────────────────────────
    print("\nComputing HERMES trend labels …")
    labels = _compute_trend_labels(history_df)

    label_counts = labels.value_counts().sort_index()
    print(f"  Declining  (-1): {label_counts.get(-1, 0):>6,}")
    print(f"  Flat        (0): {label_counts.get( 0, 0):>6,}")
    print(f"  Increasing  (1): {label_counts.get( 1, 0):>6,}")

    arima_ids = _stratified_sample(labels, n_total=n_sample, seed=seed)
    sample_counts = labels[arima_ids].value_counts().sort_index()
    print(f"\nARIMA sample ({len(arima_ids)} series, stratified by trend label):")
    for lv, name in [(-1, "declining"), (0, "flat"), (1, "increasing")]:
        print(f"  {name:<12}: {sample_counts.get(lv, 0)}")

    # ── ① Naive (all 10 000 series) ──────────────────────────────────────
    print(f"\n[1/3] Naive baseline — {n_total:,} series …")
    t0 = time.perf_counter()
    naive_rows = []
    for sid in tqdm(history_df.columns, ncols=80, leave=False):
        hist = history_df[sid].to_numpy(float)
        pred = predict_naive(hist, n_test)
        true = test_df[sid].to_numpy(float)
        naive_rows.append(_evaluate_series(true, pred, sid, "naive", int(labels[sid])))
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # ── ② Seasonal Naive (all 10 000 series) ─────────────────────────────
    print(f"\n[2/3] Seasonal naive baseline — {n_total:,} series …")
    t0 = time.perf_counter()
    snaive_rows = []
    for sid in tqdm(history_df.columns, ncols=80, leave=False):
        hist = history_df[sid].to_numpy(float)
        pred = predict_seasonal_naive(hist, n_test)
        true = test_df[sid].to_numpy(float)
        snaive_rows.append(_evaluate_series(true, pred, sid, "seasonal_naive", int(labels[sid])))
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # ── ③ ARIMA(1,1,1) (stratified 500 series) ───────────────────────────
    print(f"\n[3/3] ARIMA(1,1,1) baseline — {len(arima_ids)} stratified series …")
    t0 = time.perf_counter()
    arima_rows = []
    for sid in tqdm(arima_ids, ncols=80):
        hist = history_df[sid].to_numpy(float)
        pred = predict_arima(hist, n_test)
        true = test_df[sid].to_numpy(float)
        arima_rows.append(_evaluate_series(true, pred, sid, "arima_111", int(labels[sid])))
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # ── Combine & save ────────────────────────────────────────────────────
    results = pd.DataFrame(naive_rows + snaive_rows + arima_rows)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"\nSaved → {results_path}  ({len(results):,} rows)")

    # ── Summary tables ────────────────────────────────────────────────────
    _print_summary(results, arima_ids, labels)

    return results


# ── Reporting ──────────────────────────────────────────────────────────────

def _agg_metrics(df: pd.DataFrame, model: str) -> Dict:
    sub  = df[df["model"] == model]
    return {
        "model":      model,
        "n_series":   len(sub),
        "rmse_mean":  sub["rmse"].mean(),
        "rmse_median":sub["rmse"].median(),
        "rmse_std":   sub["rmse"].std(),
        "r2_mean":    sub["r2"].mean(),
        "r2_median":  sub["r2"].median(),
    }


def _print_summary(
    results: pd.DataFrame,
    arima_ids: List[str],
    labels: pd.Series,
) -> None:
    """Print two summary tables: full dataset and matched 500-series."""

    sep = "=" * 65

    # ── Table 1: all series available per model ──────────────────────────
    print(f"\n{sep}")
    print("BASELINE SUMMARY — full available series per model")
    print(sep)
    print(
        f"\n  {'Model':<18} {'N':>6}  {'RMSE mean':>10}  {'RMSE med':>9}"
        f"  {'R² mean':>8}  {'R² med':>8}"
    )
    print("  " + "-" * 62)
    for model in ["naive", "seasonal_naive", "arima_111"]:
        ag = _agg_metrics(results, model)
        print(
            f"  {ag['model']:<18} {ag['n_series']:>6,}  "
            f"{ag['rmse_mean']:>10.4f}  {ag['rmse_median']:>9.4f}  "
            f"{ag['r2_mean']:>8.4f}  {ag['r2_median']:>8.4f}"
        )

    # ── Table 2: matched 500-series head-to-head ─────────────────────────
    matched = results[results["series_id"].isin(arima_ids)]
    print(f"\n{sep}")
    print(f"HEAD-TO-HEAD — matched {len(arima_ids)}-series subsample (ARIMA subset)")
    print(sep)
    print(
        f"\n  {'Model':<18} {'N':>6}  {'RMSE mean':>10}  {'RMSE med':>9}"
        f"  {'R² mean':>8}  {'R² med':>8}"
    )
    print("  " + "-" * 62)
    for model in ["naive", "seasonal_naive", "arima_111"]:
        ag = _agg_metrics(matched, model)
        print(
            f"  {ag['model']:<18} {ag['n_series']:>6,}  "
            f"{ag['rmse_mean']:>10.4f}  {ag['rmse_median']:>9.4f}  "
            f"{ag['r2_mean']:>8.4f}  {ag['r2_median']:>8.4f}"
        )

    # ── Table 3: ARIMA breakdown by trend label ───────────────────────────
    arima_df = results[results["model"] == "arima_111"]
    print(f"\n{sep}")
    print("ARIMA(1,1,1) — breakdown by HERMES trend label")
    print(sep)
    print(
        f"\n  {'Trend label':<14} {'N':>6}  {'RMSE mean':>10}  {'RMSE med':>9}"
        f"  {'R² mean':>8}"
    )
    print("  " + "-" * 52)
    for lname in ["increasing", "flat", "declining"]:
        sub = arima_df[arima_df["trend_label"] == lname]
        if len(sub):
            print(
                f"  {lname:<14} {len(sub):>6}  "
                f"{sub['rmse'].mean():>10.4f}  {sub['rmse'].median():>9.4f}  "
                f"{sub['r2'].mean():>8.4f}"
            )

    # ── Table 4: SNaive breakdown by category ─────────────────────────────
    sn_df = results[results["model"] == "seasonal_naive"]
    print(f"\n{sep}")
    print("SEASONAL NAIVE — RMSE breakdown by fashion category (all series)")
    print(sep)
    print(f"\n  {'Category':<16} {'N':>6}  {'RMSE mean':>10}  {'R² mean':>8}")
    print("  " + "-" * 44)
    for cat in sorted(sn_df["category"].unique()):
        sub = sn_df[sn_df["category"] == cat]
        print(
            f"  {cat:<16} {len(sub):>6}  "
            f"{sub['rmse'].mean():>10.4f}  {sub['r2'].mean():>8.4f}"
        )

    print(f"\n{sep}")
    print("FLOOR FOR MAIN MODELS (head-to-head on 500-series subsample)")
    print(sep)
    hh_naive  = matched[matched["model"] == "naive"]["rmse"].mean()
    hh_snaive = matched[matched["model"] == "seasonal_naive"]["rmse"].mean()
    hh_arima  = matched[matched["model"] == "arima_111"]["rmse"].mean()
    floor_rmse = min(hh_naive, hh_snaive, hh_arima)
    best_name  = ["naive", "seasonal_naive", "arima_111"][
        [hh_naive, hh_snaive, hh_arima].index(floor_rmse)
    ]
    print(f"\n  Best baseline : {best_name}  (mean RMSE = {floor_rmse:.4f})")
    print(
        f"\n  XGBoost, LSTM, and TFT must each achieve mean RMSE < {floor_rmse:.4f}"
        f"\n  and mean R² > {matched[matched['model']==best_name]['r2'].mean():.4f}"
        f"\n  on the same 500-series test subsample to demonstrate improvement."
    )
    print(sep)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_baselines()
