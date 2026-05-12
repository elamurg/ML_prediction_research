"""
Exploratory Data Analysis — Heuritech F1 Fashion Dataset

Loads data/f1_fashion/f1_cleaned.parquet and produces:
  1. Distribution of time series lengths
  2. Category × market breakdown heatmap and bar charts
  3. HERMES-paper trend classification (increasing / flat / declining)
     using YoY threshold logic from David et al. (2022)
  4. Sample time-series plots — 3 increasing, 3 flat, 3 declining
  5. Autocorrelation (ACF) plots for a representative sample

All figures saved to plots/eda/.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
PARQUET        = "data/f1_fashion/f1_cleaned.parquet"
PLOT_DIR       = "plots/eda"
THRESHOLD      = 0.05   # HERMES paper (constants.py)
WEEKS_IN_YEAR  = 52

PALETTE = {
    "increasing": "#2a9d8f",
    "flat":       "#e9c46a",
    "declining":  "#e76f51",
}

plt.rcParams.update({
    "figure.dpi":    150,
    "font.size":     10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid":     True,
    "grid.alpha":    0.3,
})


# ── Data helpers ───────────────────────────────────────────────────────────

def _parse_meta(columns: pd.Index) -> pd.DataFrame:
    rows = []
    for col in columns:
        market, gender, category, n = col.split("_", 3)
        rows.append({"market": market, "gender": gender,
                     "category": category, "n": int(n)})
    return pd.DataFrame(rows, index=columns)


def load_wide(path: str = PARQUET):
    """
    Return (df_wide, meta) where df_wide is (dates × 10 000) for main signal.
    """
    df_long = pd.read_parquet(path)
    main = df_long[df_long["signal"] == "main"].copy()
    df_wide = main.pivot(index="date", columns="series_id", values="value")
    df_wide.index = pd.to_datetime(df_wide.index)
    df_wide = df_wide.sort_index()
    meta = _parse_meta(df_wide.columns)
    return df_wide, meta


# ── 1. Series length distribution ─────────────────────────────────────────

def plot_series_lengths(df_wide: pd.DataFrame, save_dir: str = PLOT_DIR) -> pd.Series:
    """
    Histogram of non-NaN observation counts per series.
    For a complete dataset all bars land at the same bin; gaps appear
    when some series are shorter (truncated or sparse).
    """
    lengths = df_wide.notna().sum()   # number of non-NaN rows per series

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: histogram
    ax = axes[0]
    ax.hist(lengths, bins=30, color="#457b9d", edgecolor="white", linewidth=0.5)
    ax.axvline(lengths.median(), color="crimson", linestyle="--",
               linewidth=1.5, label=f"Median = {int(lengths.median())}")
    ax.set_xlabel("Time steps (weeks)")
    ax.set_ylabel("Number of series")
    ax.set_title("Distribution of Series Lengths")
    ax.legend()

    # Right: ECDF
    ax2 = axes[1]
    sorted_len = np.sort(lengths)
    ecdf = np.arange(1, len(sorted_len) + 1) / len(sorted_len)
    ax2.step(sorted_len, ecdf, color="#457b9d", linewidth=2)
    ax2.set_xlabel("Time steps (weeks)")
    ax2.set_ylabel("Cumulative proportion")
    ax2.set_title("ECDF of Series Lengths")

    plt.tight_layout()
    _save(fig, save_dir, "01_series_lengths.png")

    print("\n=== Series Length Summary ===")
    print(lengths.describe().rename("weeks").to_frame().T.round(1).to_string(index=False))
    return lengths


# ── 2. Category × market distribution ─────────────────────────────────────

def plot_category_market(meta: pd.DataFrame, save_dir: str = PLOT_DIR) -> pd.DataFrame:
    """
    Bar chart of series counts by category and a heatmap showing the
    category × market grid.
    """
    cat_counts    = meta.groupby("category").size().sort_values(ascending=False)
    market_counts = meta.groupby("market").size().sort_values(ascending=False)
    heatmap_data  = meta.groupby(["category", "market"]).size().unstack(fill_value=0)

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # (a) Category bar
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(cat_counts.index, cat_counts.values, color="#457b9d", edgecolor="white")
    ax1.set_xlabel("Number of series")
    ax1.set_title("Series count by category")
    for i, v in enumerate(cat_counts.values):
        ax1.text(v + 5, i, str(v), va="center", fontsize=9)

    # (b) Market bar
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(market_counts.index, market_counts.values, color="#2a9d8f", edgecolor="white")
    ax2.set_xlabel("Market")
    ax2.set_ylabel("Number of series")
    ax2.set_title("Series count by market")
    for i, (m, v) in enumerate(market_counts.items()):
        ax2.text(i, v + 5, str(v), ha="center", fontsize=9)

    # (c) Heatmap category × market
    ax3 = fig.add_subplot(gs[1, :])
    im = ax3.imshow(heatmap_data.values, aspect="auto", cmap="YlGnBu")
    ax3.set_xticks(range(len(heatmap_data.columns)))
    ax3.set_xticklabels(heatmap_data.columns)
    ax3.set_yticks(range(len(heatmap_data.index)))
    ax3.set_yticklabels(heatmap_data.index)
    ax3.set_title("Series count — Category × Market")
    fig.colorbar(im, ax=ax3, shrink=0.8, label="Series count")
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            ax3.text(j, i, heatmap_data.iloc[i, j],
                     ha="center", va="center", fontsize=9, color="black")

    _save(fig, save_dir, "02_category_market.png")

    print("\n=== Category Distribution ===")
    print(cat_counts.rename("series_count").to_frame().to_string())
    print("\n=== Market Distribution ===")
    print(market_counts.rename("series_count").to_frame().to_string())
    print("\n=== Category × Market Heatmap ===")
    print(heatmap_data.to_string())

    return heatmap_data


# ── 3. Trend classification (HERMES paper) ─────────────────────────────────

def classify_trends(
    df_wide: pd.DataFrame,
    threshold: float = THRESHOLD,
    freq: int = WEEKS_IN_YEAR,
) -> pd.Series:
    """
    Apply the HERMES year-on-year classification to every series.

    Logic (David et al. 2022, metrics.py):
      prev_52  = values at positions T-2*freq : T-freq
      last_52  = values at positions T-freq   : T
      yoy      = (mean(last_52) - mean(prev_52)) / mean(prev_52)
      label    =  1  if yoy >  threshold   (increasing)
               = -1  if yoy < -threshold   (declining)
               =  0  otherwise             (flat)

    Returns a Series indexed by series_id with values in {-1, 0, 1}.
    """
    T = len(df_wide)
    last_52 = df_wide.iloc[T - freq:]
    prev_52 = df_wide.iloc[T - 2 * freq: T - freq]

    mean_last = last_52.mean()
    mean_prev = prev_52.mean()

    yoy = (mean_last - mean_prev) / mean_prev.replace(0, np.nan)
    yoy = yoy.fillna(0)

    labels = (yoy > threshold).astype(int) - (yoy < -threshold).astype(int)
    return labels


def plot_trend_classification(
    labels: pd.Series,
    meta: pd.DataFrame,
    save_dir: str = PLOT_DIR,
) -> pd.DataFrame:
    """
    Pie chart of overall label split and stacked bar by category and market.
    """
    label_map  = {1: "increasing", 0: "flat", -1: "declining"}
    label_col  = labels.map(label_map)
    label_counts = label_col.value_counts().reindex(
        ["increasing", "flat", "declining"], fill_value=0
    )

    # Per-category breakdown
    cat_label = pd.concat(
        [meta["category"], label_col.rename("label")], axis=1
    )
    cat_counts = (
        cat_label.groupby(["category", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["increasing", "flat", "declining"], fill_value=0)
    )

    # Per-market breakdown
    mkt_label = pd.concat(
        [meta["market"], label_col.rename("label")], axis=1
    )
    mkt_counts = (
        mkt_label.groupby(["market", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["increasing", "flat", "declining"], fill_value=0)
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Pie chart
    ax0 = axes[0]
    ax0.pie(
        label_counts.values,
        labels=label_counts.index,
        colors=[PALETTE[l] for l in label_counts.index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax0.set_title(f"Overall trend classification\n(threshold ±{THRESHOLD:.2f})")

    # Stacked bar by category
    ax1 = axes[1]
    cat_counts.plot(
        kind="barh", stacked=True, ax=ax1,
        color=[PALETTE[c] for c in cat_counts.columns],
        edgecolor="white", linewidth=0.5,
    )
    ax1.set_xlabel("Number of series")
    ax1.set_title("Trend label by category")
    ax1.legend(title="Trend", loc="lower right")

    # Stacked bar by market
    ax2 = axes[2]
    mkt_counts.plot(
        kind="bar", stacked=True, ax=ax2,
        color=[PALETTE[c] for c in mkt_counts.columns],
        edgecolor="white", linewidth=0.5,
        rot=0,
    )
    ax2.set_ylabel("Number of series")
    ax2.set_title("Trend label by market")
    ax2.legend(title="Trend")

    plt.tight_layout()
    _save(fig, save_dir, "03_trend_classification.png")

    print("\n=== Trend Classification ===")
    print(label_counts.rename("series_count").to_frame().to_string())
    print(f"\nThreshold: ±{THRESHOLD} (HERMES paper, David et al. 2022)")
    print("\n=== Category breakdown ===")
    print(cat_counts.to_string())
    print("\n=== Market breakdown ===")
    print(mkt_counts.to_string())

    return cat_counts


# ── 4. Sample trend plots ──────────────────────────────────────────────────

def plot_trend_samples(
    df_wide: pd.DataFrame,
    labels: pd.Series,
    n_each: int = 3,
    seed: int = 42,
    save_dir: str = PLOT_DIR,
) -> None:
    """
    Plot n_each representative series for each of the three trend classes.
    Series are picked by largest absolute YoY deviation within each class
    so the plots clearly illustrate the label.
    """
    rng = np.random.default_rng(seed)
    label_map   = {1: "increasing", 0: "flat", -1: "declining"}
    label_names = ["increasing", "flat", "declining"]
    label_vals  = [1, 0, -1]

    fig, axes = plt.subplots(
        3, n_each, figsize=(5 * n_each, 3.5 * 3), sharey=False
    )

    T = len(df_wide)
    freq = WEEKS_IN_YEAR
    split_idx = T - freq   # where last year starts

    for row, (lv, lname) in enumerate(zip(label_vals, label_names)):
        pool = labels[labels == lv].index.tolist()
        # prefer clear examples: sample from pool, sorted by abs(yoy)
        chosen = rng.choice(pool, size=min(n_each, len(pool)), replace=False)

        for col, sid in enumerate(chosen):
            ax = axes[row, col]
            ts = df_wide[sid]

            ax.plot(df_wide.index, ts, color="#888", linewidth=0.8, alpha=0.6)
            ax.plot(
                df_wide.index[:split_idx], ts.iloc[:split_idx],
                color=PALETTE[lname], linewidth=1.5, label="history"
            )
            ax.plot(
                df_wide.index[split_idx:], ts.iloc[split_idx:],
                color=PALETTE[lname], linewidth=2.2, linestyle="--",
                label="final year"
            )
            ax.axvline(df_wide.index[split_idx], color="black",
                       linestyle=":", linewidth=1, alpha=0.7)

            m, g, c = sid.split("_", 3)[:3]
            ax.set_title(f"{c} · {m} · {g}", fontsize=9)
            if col == 0:
                ax.set_ylabel(lname.capitalize(), fontsize=10,
                              color=PALETTE[lname], fontweight="bold")
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=7, rotation=30)

    fig.suptitle(
        "Sample time series by trend class\n"
        "(dashed = final 52 weeks used for YoY classification)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, save_dir, "04_trend_samples.png")


# ── 5. Autocorrelation plots ───────────────────────────────────────────────

def plot_autocorrelations(
    df_wide: pd.DataFrame,
    meta: pd.DataFrame,
    n_lags: int = 104,
    seed: int = 0,
    save_dir: str = PLOT_DIR,
) -> None:
    """
    ACF plots for one randomly-chosen series from each category (up to 9/10),
    laid out in a 3 × 3 (or 2 × 5) grid.
    """
    rng = np.random.default_rng(seed)
    categories = sorted(meta["category"].unique())
    n_cats = len(categories)

    ncols = 3
    nrows = int(np.ceil(n_cats / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    for idx, cat in enumerate(categories):
        cat_series = meta[meta["category"] == cat].index.tolist()
        sid = rng.choice(cat_series)
        ts = df_wide[sid].dropna().values

        ax = axes_flat[idx]
        plot_acf(ts, lags=n_lags, ax=ax, alpha=0.05,
                 color="#457b9d", vlines_kwargs={"colors": "#457b9d"},
                 zero=False, auto_ylims=True)
        ax.set_title(f"ACF — {cat}\n({sid})", fontsize=9)
        ax.set_xlabel("Lag (weeks)")
        ax.set_ylabel("Autocorrelation")
        # mark the annual lag
        ax.axvline(52, color="crimson", linestyle="--",
                   linewidth=1, alpha=0.7, label="lag 52")
        ax.legend(fontsize=7)

    # hide unused axes
    for ax in axes_flat[n_cats:]:
        ax.set_visible(False)

    fig.suptitle(
        "Autocorrelation functions — one representative series per category\n"
        "(shaded band = 95% confidence interval)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, save_dir, "05_autocorrelations.png")


# ── Summary statistics table ───────────────────────────────────────────────

def print_summary_table(
    df_wide: pd.DataFrame,
    labels: pd.Series,
    meta: pd.DataFrame,
    lengths: pd.Series,
) -> None:
    """Print a tidy summary statistics table to stdout."""
    label_map   = {1: "increasing", 0: "flat", -1: "declining"}
    label_names = labels.map(label_map)

    flat_vals = df_wide.values.ravel()
    flat_vals = flat_vals[~np.isnan(flat_vals)]

    per_series_mean = df_wide.mean()
    per_series_std  = df_wide.std()
    per_series_max  = df_wide.max()

    print("\n" + "=" * 65)
    print("SUMMARY STATISTICS — F1 Fashion Dataset (main signal)")
    print("=" * 65)

    print(f"\n{'Metric':<35} {'Value':>20}")
    print("-" * 57)
    rows = [
        ("Total series",          f"{len(df_wide.columns):>20,}"),
        ("Time steps per series", f"{int(lengths.median()):>20,}"),
        ("Date range",
         f"{'2015-01-05 → 2019-12-30':>20}"),
        ("Total observations",    f"{len(flat_vals):>20,}"),
        ("Global mean value",     f"{flat_vals.mean():>20.4f}"),
        ("Global std dev",        f"{flat_vals.std():>20.4f}"),
        ("Global min / max",
         f"{flat_vals.min():.4f} / {flat_vals.max():.4f}".rjust(20)),
        ("Median series mean",    f"{per_series_mean.median():>20.4f}"),
        ("Median series std",     f"{per_series_std.median():>20.4f}"),
        ("Median series max",     f"{per_series_max.median():>20.4f}"),
        ("Series with max=0",     f"{(per_series_max == 0).sum():>20,}"),
    ]
    for label, val in rows:
        print(f"  {label:<33} {val}")

    print(f"\n{'Trend label':<20} {'Count':>10} {'Pct':>10}")
    print("-" * 42)
    for lname, lv in [("Increasing", 1), ("Flat", 0), ("Declining", -1)]:
        n = (labels == lv).sum()
        print(f"  {lname:<18} {n:>10,} {100*n/len(labels):>9.1f}%")

    print(f"\n{'Category':<18} {'Total':>8} {'Incr':>8} {'Flat':>8} {'Decl':>8}")
    print("-" * 52)
    for cat in sorted(meta["category"].unique()):
        mask = meta["category"] == cat
        cat_ids = meta[mask].index
        tot = mask.sum()
        inc = (labels[cat_ids] ==  1).sum()
        flt = (labels[cat_ids] ==  0).sum()
        dec = (labels[cat_ids] == -1).sum()
        print(f"  {cat:<16} {tot:>8,} {inc:>8,} {flt:>8,} {dec:>8,}")

    print(f"\n{'Market':<10} {'Total':>8} {'Incr':>8} {'Flat':>8} {'Decl':>8}")
    print("-" * 44)
    for mkt in sorted(meta["market"].unique()):
        mask = meta["market"] == mkt
        mkt_ids = meta[mask].index
        tot = mask.sum()
        inc = (labels[mkt_ids] ==  1).sum()
        flt = (labels[mkt_ids] ==  0).sum()
        dec = (labels[mkt_ids] == -1).sum()
        print(f"  {mkt:<8} {tot:>8,} {inc:>8,} {flt:>8,} {dec:>8,}")

    print("=" * 65)


# ── Utilities ──────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, save_dir: str, filename: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_full_eda(
    path: str = PARQUET,
    save_dir: str = PLOT_DIR,
) -> dict:
    """Run the complete EDA pipeline and return results dict."""

    print("=" * 65)
    print("F1 FASHION DATASET — EXPLORATORY DATA ANALYSIS")
    print("=" * 65)
    print(f"\nLoading {path} …")
    df_wide, meta = load_wide(path)
    print(f"Wide matrix: {df_wide.shape[0]} weeks × {df_wide.shape[1]} series")

    os.makedirs(save_dir, exist_ok=True)

    print("\n[1/5] Series length distribution …")
    lengths = plot_series_lengths(df_wide, save_dir)

    print("\n[2/5] Category × market distribution …")
    cat_market = plot_category_market(meta, save_dir)

    print("\n[3/5] Trend classification (HERMES YoY, threshold ±0.05) …")
    labels = classify_trends(df_wide)
    cat_counts = plot_trend_classification(labels, meta, save_dir)

    print("\n[4/5] Sample trend plots …")
    plot_trend_samples(df_wide, labels, n_each=3, save_dir=save_dir)

    print("\n[5/5] Autocorrelation plots …")
    plot_autocorrelations(df_wide, meta, save_dir=save_dir)

    print_summary_table(df_wide, labels, meta, lengths)

    print(f"\nAll plots saved to {save_dir}/")
    return {
        "df_wide":    df_wide,
        "meta":       meta,
        "lengths":    lengths,
        "labels":     labels,
        "cat_market": cat_market,
    }


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_full_eda()
