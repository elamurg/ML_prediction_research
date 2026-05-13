"""
Train / Validation / Test Split — Heuritech F1 Fashion Dataset

Split strategy: strict temporal ordering
  Train : first 70 % of timesteps  (indices [0,       train_end)  )
  Val   : next  15 % of timesteps  (indices [train_end, val_end)  )
  Test  : final 15 % of timesteps  (indices [val_end,   T)        )

The same index boundaries are applied to every one of the 10 000 series.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY RANDOM SPLITS CAUSE DATA LEAKAGE IN TIME SERIES FORECASTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A random split treats each observation as i.i.d. — independent and
identically distributed — and shuffles them before partitioning.
This assumption breaks down completely for time series for four reasons:

1. TEMPORAL AUTOCORRELATION
   Fashion time series are strongly autocorrelated: value at week t is
   correlated with values at weeks t-1, t-2, …  A random split can put
   week t in training and week t+1 in test, so the model has already
   "seen" the neighbourhood of every test observation during training.
   The reported test error will be far lower than any real deployment
   error — a classic optimistic-bias / data-leakage scenario.

2. LOOK-AHEAD BIAS IN LAGGED FEATURES
   Models for time series forecasting typically use lagged features
   (x_{t-1}, x_{t-2}, …) or rolling statistics.  If t is in the test
   set but t+1 is in training, computing x_{t+1}'s lag-1 feature
   requires x_t — a test observation — at training time.  The model
   implicitly learns the test set distribution during training.

3. NORMALISATION / SCALING LEAKAGE
   When scaling parameters (mean, standard deviation, min/max) are
   computed over a randomly-mixed training set, future observations
   contribute to those statistics.  Any subsequent normalisation of
   the test set therefore encodes information that would be unavailable
   at real deployment time.

4. SEASONAL / TREND LEAKAGE
   The F1 dataset spans 2015–2019 and shows annual seasonality
   (period = 52 weeks).  A random split scatters every season across
   both train and test, so the model is effectively trained on the full
   seasonal cycle before evaluation.  A temporal split forces the model
   to generalise to a future season it has never seen — the correct
   evaluation protocol for fashion trend forecasting.

The only correct protocol is: train strictly on the past, evaluate
strictly on the future.  Our 70 / 15 / 15 temporal split enforces this:

  ┌────────────────────────────────────────────────────────────────┐
  │  TRAIN (70 %)  │  VAL (15 %)  │  TEST (15 %)                  │
  │  fit models    │  tune HPs    │  final evaluation              │
  │  2015-01-05    │  2018-07-02  │  2019-04-14                   │
  │       →        │      →       │      →  2019-12-30            │
  └────────────────────────────────────────────────────────────────┘

References:
  • Hyndman & Athanasopoulos, "Forecasting: Principles and Practice",
    §3.4 Evaluating forecast accuracy — time-series cross-validation.
  • Bergmeir & Benítez (2012), "On the use of cross-validation for
    time series predictor evaluation", Information Sciences 191.
  • David et al. (2022), "HERMES: Hybrid Error-corrector Model with
    inclusion of External Signal for non-stationary time series",
    ICASSP 2022.  (The F1 dataset paper; uses 3-year train / 1-year
    test — a strict temporal holdout.)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import os
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import pandas as pd

# ── Paths & ratios ─────────────────────────────────────────────────────────
PARQUET     = "data/f1_fashion/f1_cleaned.parquet"
SPLITS_JSON = "data/f1_fashion/splits.json"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15      # = 1 - TRAIN_RATIO - VAL_RATIO

# module-level cache so get_split() avoids repeated I/O
_df_wide: pd.DataFrame | None = None


# ── Internal helpers ───────────────────────────────────────────────────────

def _load_wide(parquet_path: str = PARQUET) -> pd.DataFrame:
    """
    Load f1_cleaned.parquet, pivot to wide format (dates × series_id),
    and cache the result at module level.
    Only the 'main' signal is used for model training/evaluation.
    """
    global _df_wide
    if _df_wide is None:
        df_long = pd.read_parquet(parquet_path)
        main = df_long[df_long["signal"] == "main"]
        wide = main.pivot(index="date", columns="series_id", values="value")
        wide.index = pd.to_datetime(wide.index)
        _df_wide = wide.sort_index()
    return _df_wide


def _split_indices(T: int) -> Tuple[int, int]:
    """
    Compute (train_end, val_end) integer indices for T total timesteps.

    Indices are computed with integer truncation so that:
      train : [0, train_end)
      val   : [train_end, val_end)
      test  : [val_end, T)

    Remainder timesteps (due to truncation) fall into the test set,
    meaning the test set is never shorter than intended.
    """
    train_end = int(T * TRAIN_RATIO)
    val_end   = int(T * (TRAIN_RATIO + VAL_RATIO))
    return train_end, val_end


# ── Public API ─────────────────────────────────────────────────────────────

def compute_and_save_splits(
    parquet_path: str = PARQUET,
    splits_path:  str = SPLITS_JSON,
) -> dict:
    """
    Compute split boundaries, print date ranges, and write splits.json.

    The split is defined entirely by two integer indices (train_end,
    val_end) that are identical for every series — guaranteeing that
    no series leaks future information into an earlier split.

    Returns the metadata dict that was written to disk.
    """
    df_wide = _load_wide(parquet_path)

    T          = len(df_wide)
    dates      = df_wide.index
    series_ids = df_wide.columns.tolist()

    train_end, val_end = _split_indices(T)

    train_n = train_end
    val_n   = val_end - train_end
    test_n  = T - val_end

    train_dates = dates[:train_end]
    val_dates   = dates[train_end:val_end]
    test_dates  = dates[val_end:]

    # ── Console output ───────────────────────────────────────────────────
    print("=" * 65)
    print("TEMPORAL TRAIN / VALIDATION / TEST SPLIT — F1 Fashion Dataset")
    print("=" * 65)

    print(f"\n  Total timesteps : {T:>6}  ({dates[0].date()} → {dates[-1].date()})")
    print(f"  Total series    : {len(series_ids):>6}")
    print()

    _print_split_block("TRAIN", train_n, T, train_dates[0], train_dates[-1],
                       0, train_end - 1, TRAIN_RATIO)
    _print_split_block("VAL  ", val_n,   T, val_dates[0],   val_dates[-1],
                       train_end, val_end - 1, VAL_RATIO)
    _print_split_block("TEST ", test_n,  T, test_dates[0],  test_dates[-1],
                       val_end, T - 1, TEST_RATIO)

    print()
    print("  Split indices (0-based, half-open intervals):")
    print(f"    train : [0, {train_end})")
    print(f"    val   : [{train_end}, {val_end})")
    print(f"    test  : [{val_end}, {T})")

    # ── Build metadata ───────────────────────────────────────────────────
    meta = {
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val":   VAL_RATIO,
            "test":  TEST_RATIO,
        },
        "total_timesteps": T,
        "series_count":    len(series_ids),
        "indices": {
            "train_start": 0,
            "train_end":   train_end,       # exclusive upper bound
            "val_start":   train_end,
            "val_end":     val_end,         # exclusive upper bound
            "test_start":  val_end,
            "test_end":    T,               # exclusive upper bound
        },
        "timestep_counts": {
            "train": train_n,
            "val":   val_n,
            "test":  test_n,
        },
        "date_ranges": {
            "train": {
                "start": str(train_dates[0].date()),
                "end":   str(train_dates[-1].date()),
            },
            "val": {
                "start": str(val_dates[0].date()),
                "end":   str(val_dates[-1].date()),
            },
            "test": {
                "start": str(test_dates[0].date()),
                "end":   str(test_dates[-1].date()),
            },
        },
        "all_dates": [str(d.date()) for d in dates],
        "series_ids": series_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Strict temporal split: train on past, validate on present, "
            "test on future.  All indices are 0-based half-open intervals. "
            "Apply identically across all series — never shuffle time."
        ),
    }

    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    size_kb = os.path.getsize(splits_path) / 1024
    print(f"\n  Saved → {splits_path}  ({size_kb:.0f} KB)")

    return meta


def get_split(
    series_id: str,
    parquet_path: str = PARQUET,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (train, val, test) value arrays for a single series.

    Parameters
    ----------
    series_id   : column name in the wide matrix, e.g. 'us_female_outerwear_0'
    parquet_path: path to f1_cleaned.parquet (uses module cache after first call)

    Returns
    -------
    train : np.ndarray shape (train_n,)   — first 70 % of timesteps
    val   : np.ndarray shape (val_n,)     — next  15 % of timesteps
    test  : np.ndarray shape (test_n,)    — final 15 % of timesteps

    Example
    -------
    >>> train, val, test = get_split("us_female_outerwear_0")
    >>> print(train.shape, val.shape, test.shape)
    (182,) (39,) (40,)
    """
    df_wide = _load_wide(parquet_path)

    if series_id not in df_wide.columns:
        available = df_wide.columns[:5].tolist()
        raise KeyError(
            f"'{series_id}' not in dataset.  "
            f"Example valid IDs: {available}"
        )

    T = len(df_wide)
    train_end, val_end = _split_indices(T)

    values = df_wide[series_id].to_numpy(dtype=float)
    return values[:train_end], values[train_end:val_end], values[val_end:]


def get_split_dates(parquet_path: str = PARQUET) -> dict:
    """
    Return the exact pd.DatetimeIndex for each split as a dict.

    Useful when building date-aware tensors for sequence models.
    """
    df_wide = _load_wide(parquet_path)
    T = len(df_wide)
    train_end, val_end = _split_indices(T)
    dates = df_wide.index
    return {
        "train": dates[:train_end],
        "val":   dates[train_end:val_end],
        "test":  dates[val_end:],
    }


def get_all_splits(parquet_path: str = PARQUET) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (train_df, val_df, test_df) wide DataFrames for all series at once.

    Shape of each: (split_timesteps, 10 000).
    Preferable when vectorising operations over the full dataset.
    """
    df_wide = _load_wide(parquet_path)
    T = len(df_wide)
    train_end, val_end = _split_indices(T)
    return df_wide.iloc[:train_end], df_wide.iloc[train_end:val_end], df_wide.iloc[val_end:]


# ── Formatting helper ──────────────────────────────────────────────────────

def _print_split_block(name, n, T, start, end, idx_start, idx_end, ratio):
    pct = 100 * n / T
    print(
        f"  {name}  {start.date()}  →  {end.date()}"
        f"  |  {n:>3} weeks  ({pct:.1f} %)  "
        f"  indices [{idx_start}, {idx_end}]"
    )


# ── Summary table ──────────────────────────────────────────────────────────

def print_split_summary(parquet_path: str = PARQUET) -> None:
    """Print a compact summary of the split sizes for all 10 000 series."""
    df_wide = _load_wide(parquet_path)
    T = len(df_wide)
    train_end, val_end = _split_indices(T)

    train_n = train_end
    val_n   = val_end - train_end
    test_n  = T - val_end

    print("\n" + "=" * 55)
    print("SPLIT SUMMARY")
    print("=" * 55)
    print(f"  {'Split':<8} {'Weeks':>6}  {'Pct':>6}  {'Start':>12}  {'End':>12}")
    print("-" * 55)
    dates = df_wide.index
    for label, n, s, e in [
        ("train", train_n, dates[0],          dates[train_end - 1]),
        ("val",   val_n,   dates[train_end],   dates[val_end   - 1]),
        ("test",  test_n,  dates[val_end],     dates[-1]),
    ]:
        print(
            f"  {label:<8} {n:>6}  {100*n/T:>5.1f}%"
            f"  {str(s.date()):>12}  {str(e.date()):>12}"
        )
    print("=" * 55)
    print(f"  {'TOTAL':<8} {T:>6}  100.0%")
    print(f"\n  Applied identically across all {len(df_wide.columns):,} series.")
    print("=" * 55)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    meta = compute_and_save_splits()
    print_split_summary()

    # Demonstrate get_split() on a few representative series
    print("\n" + "=" * 65)
    print("get_split() DEMO — shape check across series types")
    print("=" * 65)

    demo_ids = [
        "us_female_outerwear_0",
        "fr_male_denim_42",
        "cn_female_dresses_99",
        "br_male_sportswear_7",
        "uk_female_bags_50",
    ]
    print(f"\n  {'series_id':<35} {'train':>7} {'val':>5} {'test':>6}")
    print("  " + "-" * 57)
    for sid in demo_ids:
        tr, va, te = get_split(sid)
        print(f"  {sid:<35} {tr.shape[0]:>7} {va.shape[0]:>5} {te.shape[0]:>6}")

    # Verify consistency: all series must have identical split sizes
    print("\n  Verifying split consistency across all 10 000 series …")
    train_df, val_df, test_df = get_all_splits()
    assert train_df.shape[1] == val_df.shape[1] == test_df.shape[1] == 10_000
    assert train_df.shape[0] + val_df.shape[0] + test_df.shape[0] == len(
        _load_wide()
    ), "Split sizes do not sum to total timesteps"
    print(
        f"  OK — train {train_df.shape}, val {val_df.shape}, test {test_df.shape}"
    )
    print(
        f"  All {train_df.shape[0] + val_df.shape[0] + test_df.shape[0]} timesteps accounted for."
    )
