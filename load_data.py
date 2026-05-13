"""
Data Loading and Cleaning – Heuritech F1 Fashion Dataset

Loads 10 000 weekly fashion time series from the F1 dataset published in:

  "HERMES: Hybrid Error-corrector Model with inclusion of External Signal
   for non-stationary time series" (David et al., 2022)
  https://arxiv.org/abs/2202.03224

Dataset files expected under data/f1_fashion/:
  f1_main.csv            – 10 000 normalized weekly fashion trend series
  f1_fashion_forward.csv – matching external weak signals (influencer behaviour)

Column naming convention: {market}_{gender}_{category}_{n}
  e.g. us_female_outerwear_0

NOTE: The original distribution URL
  http://files.heuritech.com/raw_files/f1_fashion_dataset.tar.xz
returned HTTP 404 as of 2026.  The data/f1_fashion/ directory contains a
synthetic replica produced in the exact documented format (same taxonomy,
same date range 2015-01-05 – 2019-12-30, same column naming convention).
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Documented dataset constants ───────────────────────────────────────────
WEEKS_PER_SERIES = 261          # 2015-01-05 → 2019-12-30
N_SERIES = 10_000
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "f1_fashion")
MAIN_CSV = os.path.join(DATA_DIR, "f1_main.csv")
FF_CSV   = os.path.join(DATA_DIR, "f1_fashion_forward.csv")
PARQUET  = os.path.join(DATA_DIR, "f1_cleaned.parquet")


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_column_meta(columns: pd.Index) -> pd.DataFrame:
    """Return a DataFrame with market/gender/category/n parsed from column names."""
    rows = []
    for col in columns:
        market, gender, category, n = col.split("_", 3)
        rows.append({"market": market, "gender": gender, "category": category, "n": int(n)})
    return pd.DataFrame(rows, index=columns)


def _print_breakdown(meta: pd.DataFrame) -> None:
    print("\n--- Market breakdown ---")
    market_counts = meta.groupby("market").size().sort_values(ascending=False)
    for market, count in market_counts.items():
        print(f"  {market:<6} {count:>6} series")

    print("\n--- Gender breakdown ---")
    gender_counts = meta.groupby("gender").size().sort_values(ascending=False)
    for gender, count in gender_counts.items():
        print(f"  {gender:<10} {count:>6} series")

    print("\n--- Category breakdown ---")
    cat_counts = meta.groupby("category").size().sort_values(ascending=False)
    for cat, count in cat_counts.items():
        print(f"  {cat:<15} {count:>6} series")

    print("\n--- Market × Gender breakdown ---")
    mg_counts = meta.groupby(["market", "gender"]).size().unstack(fill_value=0)
    print(mg_counts.to_string())


# ── Public API ─────────────────────────────────────────────────────────────

def load_f1_main(filepath: str = MAIN_CSV) -> pd.DataFrame:
    """
    Load f1_main.csv.

    Returns a DataFrame with shape (261, 10 000):
      - index : weekly dates (str, YYYY-MM-DD)
      - columns: time series names {market}_{gender}_{category}_{n}
    """
    print("=" * 60)
    print("LOADING F1 MAIN FASHION TIME SERIES")
    print("=" * 60)
    print(f"\nSource: {filepath}")

    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    print(f"\nShape: {df.shape}  ({df.shape[1]} series × {df.shape[0]} time steps)")
    print(f"Time steps per series: {df.shape[0]}")
    print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"\nMissing values: {df.isna().sum().sum()}")
    print(f"Value range : [{df.values.min():.4f}, {df.values.max():.4f}]")

    return df


def load_f1_fashion_forward(filepath: str = FF_CSV) -> pd.DataFrame:
    """
    Load f1_fashion_forward.csv (external weak signals / influencer behaviour).

    Returns the same shape and structure as load_f1_main().
    """
    print("\n" + "=" * 60)
    print("LOADING F1 FASHION-FORWARD (EXTERNAL SIGNALS)")
    print("=" * 60)
    print(f"\nSource: {filepath}")

    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    print(f"\nShape: {df.shape}  ({df.shape[1]} series × {df.shape[0]} time steps)")
    print(f"Time steps per series: {df.shape[0]}")
    print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"\nMissing values: {df.isna().sum().sum()}")
    print(f"Value range : [{df.values.min():.4f}, {df.values.max():.4f}]")

    return df


def load_all_f1_data(
    main_path: str = MAIN_CSV,
    ff_path: str = FF_CSV,
    save_parquet: bool = True,
    parquet_path: str = PARQUET,
) -> dict:
    """
    Load the complete F1 fashion dataset, print diagnostics, and optionally
    save a cleaned long-form DataFrame to Parquet.

    Parameters
    ----------
    main_path    : path to f1_main.csv
    ff_path      : path to f1_fashion_forward.csv
    save_parquet : whether to write data/f1_fashion/f1_cleaned.parquet
    parquet_path : destination path for the Parquet file

    Returns
    -------
    dict with keys:
      'main'     – wide DataFrame (261 × 10 000) from f1_main.csv
      'forward'  – wide DataFrame (261 × 10 000) from f1_fashion_forward.csv
      'meta'     – per-series metadata (market, gender, category, n)
      'long'     – long-form DataFrame with columns
                   [date, series_id, market, gender, category, n,
                    value, signal]  (saved to Parquet)
    """
    df_main = load_f1_main(main_path)
    df_ff   = load_f1_fashion_forward(ff_path)

    # ── Metadata ────────────────────────────────────────────────────────────
    meta = _parse_column_meta(df_main.columns)

    print("\n" + "=" * 60)
    print("CATEGORY AND MARKET BREAKDOWN")
    print("=" * 60)
    _print_breakdown(meta)

    # ── Build cleaned long-form DataFrame ────────────────────────────────────
    print("\n" + "=" * 60)
    print("BUILDING CLEANED LONG-FORM DATAFRAME")
    print("=" * 60)

    # Melt main series
    long_main = (
        df_main.reset_index()
        .melt(id_vars="date", var_name="series_id", value_name="value")
    )
    long_main["signal"] = "main"

    # Melt forward signals
    long_ff = (
        df_ff.reset_index()
        .melt(id_vars="date", var_name="series_id", value_name="value")
    )
    long_ff["signal"] = "fashion_forward"

    long = pd.concat([long_main, long_ff], ignore_index=True)

    # Join metadata
    long = long.merge(
        meta[["market", "gender", "category", "n"]].reset_index().rename(
            columns={"index": "series_id"}
        ),
        on="series_id",
        how="left",
    )
    long = long[["date", "series_id", "market", "gender", "category", "n", "value", "signal"]]
    long = long.sort_values(["signal", "series_id", "date"]).reset_index(drop=True)

    print(f"\nLong-form shape : {long.shape}")
    print(f"Columns         : {long.columns.tolist()}")
    print(f"Unique series   : {long['series_id'].nunique()}")
    print(f"Unique dates    : {long['date'].nunique()}")
    print(f"\nSample rows:")
    print(long.head(6).to_string(index=False))

    # ── Save parquet ─────────────────────────────────────────────────────────
    if save_parquet:
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        long.to_parquet(parquet_path, index=False)
        size_mb = os.path.getsize(parquet_path) / 1e6
        print(f"\nSaved cleaned DataFrame → {parquet_path}  ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE")
    print("=" * 60)

    return {
        "main":    df_main,
        "forward": df_ff,
        "meta":    meta,
        "long":    long,
    }


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_all_f1_data()

    print("\n\n--- Wide main (first 3 cols, 5 rows) ---")
    print(data["main"].iloc[:5, :3])

    print("\n--- Wide fashion-forward (first 3 cols, 5 rows) ---")
    print(data["forward"].iloc[:5, :3])

    print("\n--- Metadata sample ---")
    print(data["meta"].head(8))
