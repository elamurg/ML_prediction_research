"""
XGBoost Model — Heuritech F1 Fashion Dataset

One XGBoost model is trained per series on the 221-week history (train + val),
then evaluated on the 40-week test split using recursive multi-step forecasting.

Applied to the same 500-series stratified sample as baselines.py:
  • HERMES trend labels (increasing / flat / declining), seed = 42
  • ~167 declining / ~167 flat / ~166 increasing

Feature set (17 features, zero data-leakage by construction):
  ┌──────────────────┬────────────────────────────────────────────────┐
  │ Group            │ Features                                       │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ Lag              │ lag_1, lag_2, lag_4, lag_8, lag_12             │
  │ Rolling          │ rmean_4/8/12, rstd_4/8/12  (over t-w … t-1)  │
  │ Momentum         │ mom_1 = v[t-1]-v[t-2],  mom_4 = v[t-1]-v[t-5]│
  │ Cyclical calendar│ sin/cos of week-of-year and month-of-year     │
  └──────────────────┴────────────────────────────────────────────────┘

All features at step t are computed exclusively from values observed
before t (i.e. history[0 … t-1]), so no look-ahead leakage is possible.
Recursive forecasting extends this discipline to the test horizon:
at each test step the previous step's prediction is appended to the
buffer before building the next feature row.

SHAP analysis on a 50-series subsample → plots/shap/
Results → results/xgboost_results.csv
"""

import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths & constants ──────────────────────────────────────────────────────
PARQUET     = "data/f1_fashion/f1_cleaned.parquet"
SPLITS_JSON = "data/f1_fashion/splits.json"
RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "xgboost_results.csv")
SHAP_DIR    = "plots/shap"

SEED      = 42
N_SAMPLE  = 500
N_SHAP    = 50
FREQ      = 52
THRESHOLD = 0.05

# ── Feature specification ──────────────────────────────────────────────────
LAG_FEATS      = ["lag_1", "lag_2", "lag_4", "lag_8", "lag_12"]
ROLLING_FEATS  = ["rmean_4", "rmean_8", "rmean_12",
                  "rstd_4",  "rstd_8",  "rstd_12"]
MOMENTUM_FEATS = ["mom_1", "mom_4"]
CALENDAR_FEATS = ["sin_week", "cos_week", "sin_month", "cos_month"]
FEATURE_NAMES  = LAG_FEATS + ROLLING_FEATS + MOMENTUM_FEATS + CALENDAR_FEATS

FEATURE_GROUPS = {
    "lag":      LAG_FEATS,
    "rolling":  ROLLING_FEATS,
    "momentum": MOMENTUM_FEATS,
    "calendar": CALENDAR_FEATS,
}

# ── XGBoost hyperparameters ────────────────────────────────────────────────
# Conservative depth (3) and high shrinkage (lr=0.05) to guard against
# overfitting on the ~209-row training window per series.
XGB_PARAMS = dict(
    objective        = "reg:squarederror",
    n_estimators     = 300,
    max_depth        = 3,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 2,
    random_state     = SEED,
    verbosity        = 0,
    tree_method      = "hist",   # fast CPU solver
)


# ── Data helpers ───────────────────────────────────────────────────────────

def _load_data():
    """Return (history_df, test_df, meta) wide DataFrames."""
    with open(SPLITS_JSON) as fh:
        meta = json.load(fh)

    idx     = meta["indices"]
    df_long = pd.read_parquet(PARQUET)
    main    = df_long[df_long["signal"] == "main"]
    wide    = main.pivot(index="date", columns="series_id", values="value")
    wide.index = pd.to_datetime(wide.index)
    wide    = wide.sort_index()

    train_df  = wide.iloc[idx["train_start"] : idx["train_end"]]
    val_df    = wide.iloc[idx["val_start"]   : idx["val_end"]]
    test_df   = wide.iloc[idx["test_start"]  : idx["test_end"]]
    history_df = pd.concat([train_df, val_df])   # 221 weeks
    return history_df, test_df, meta


def _series_meta(series_id: str) -> dict:
    market, gender, category, n = series_id.split("_", 3)
    return dict(market=market, gender=gender, category=category, n=int(n))


# ── HERMES trend labels & stratified sample ────────────────────────────────

def _compute_trend_labels(history_df: pd.DataFrame) -> pd.Series:
    """YoY classification: 1=increasing, 0=flat, -1=declining."""
    last = history_df.iloc[-FREQ:]
    prev = history_df.iloc[-2 * FREQ : -FREQ]
    yoy  = ((last.mean() - prev.mean()) / prev.mean().replace(0, np.nan)).fillna(0)
    return (yoy > THRESHOLD).astype(int) - (yoy < -THRESHOLD).astype(int)


def _stratified_sample(labels: pd.Series, n: int = N_SAMPLE,
                       seed: int = SEED) -> list:
    """Balanced draw across HERMES trend classes (same logic as baselines.py)."""
    rng     = np.random.default_rng(seed)
    classes = sorted(labels.unique())
    base    = n // len(classes)
    extra   = n  % len(classes)
    sampled = []
    for i, cls in enumerate(classes):
        pool   = labels[labels == cls].index.tolist()
        k      = min(base + (1 if i < extra else 0), len(pool))
        sampled.extend(rng.choice(pool, size=k, replace=False).tolist())
    return sampled


# ── Feature engineering ────────────────────────────────────────────────────

def _cyclical_calendar(dates: pd.DatetimeIndex) -> dict:
    """Return sin/cos encodings for week-of-year and month."""
    week = dates.isocalendar().week.astype(float).values
    mon  = dates.month.astype(float).values
    return {
        "sin_week":  np.sin(2 * np.pi * week / 52),
        "cos_week":  np.cos(2 * np.pi * week / 52),
        "sin_month": np.sin(2 * np.pi * mon  / 12),
        "cos_month": np.cos(2 * np.pi * mon  / 12),
    }


def build_train_features(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construct (X, y) for all positions in `values`, then drop NaN rows.

    Features for predicting value[t] are built from value[0 … t-1]:
      • lag_k        = value[t-k]
      • rmean_w      = mean(value[t-w … t-1])   (shift-1 then rolling)
      • rstd_w       = std (value[t-w … t-1])
      • mom_1        = value[t-1] − value[t-2]
      • mom_4        = value[t-1] − value[t-5]
      • calendar     = sin/cos of ISO week and calendar month

    The first 12 rows are dropped because lag_12 is undefined there.
    All rolling windows use shift(1) first so they never touch the
    current-step value being predicted.
    """
    s = pd.Series(values, index=dates)
    f = pd.DataFrame(index=dates)

    for k in [1, 2, 4, 8, 12]:
        f[f"lag_{k}"] = s.shift(k)

    s1 = s.shift(1)   # values ending at t-1 (no leakage)
    for w in [4, 8, 12]:
        f[f"rmean_{w}"] = s1.rolling(w, min_periods=1).mean()
        f[f"rstd_{w}"]  = s1.rolling(w, min_periods=2).std().fillna(0.0)

    f["mom_1"] = s.shift(1) - s.shift(2)
    f["mom_4"] = s.shift(1) - s.shift(5)

    cal = _cyclical_calendar(dates)
    for name, arr in cal.items():
        f[name] = arr

    f["_target"] = values
    f = f.dropna(subset=FEATURE_NAMES)   # drops first 12 rows

    return f[FEATURE_NAMES], f["_target"]


def _feature_row(buf: list, t: int, date: pd.Timestamp) -> np.ndarray:
    """
    Compute a single feature row for recursive forecasting.

    `buf` contains all values observed up to (but not including) position t;
    `date` is the calendar date at position t.

    Exactly mirrors the pandas version in build_train_features so models
    see the same feature distribution during prediction as during training.
    """
    row = np.empty(len(FEATURE_NAMES), dtype=float)
    i = 0

    # Lag features
    for k in [1, 2, 4, 8, 12]:
        row[i] = buf[t - k] if t - k >= 0 else np.nan
        i += 1

    # Rolling mean (last w values ending at t-1)
    for w in [4, 8, 12]:
        window = buf[max(0, t - w): t]
        row[i] = float(np.mean(window)) if window else np.nan
        i += 1

    # Rolling std (last w values ending at t-1)
    for w in [4, 8, 12]:
        window = buf[max(0, t - w): t]
        row[i] = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
        i += 1

    # Momentum
    row[i]     = buf[t-1] - buf[t-2] if t >= 2 else np.nan;  i += 1
    row[i]     = buf[t-1] - buf[t-5] if t >= 5 else np.nan;  i += 1

    # Cyclical calendar
    ts   = pd.Timestamp(date)
    week = float(ts.isocalendar()[1])
    mon  = float(ts.month)
    row[i] = np.sin(2 * np.pi * week / 52);  i += 1
    row[i] = np.cos(2 * np.pi * week / 52);  i += 1
    row[i] = np.sin(2 * np.pi * mon  / 12);  i += 1
    row[i] = np.cos(2 * np.pi * mon  / 12)

    return row


# ── Training & recursive forecasting ──────────────────────────────────────

def train_xgb(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """Fit an XGBRegressor on the provided feature / target matrix."""
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y, verbose=False)
    return model


def recursive_forecast(
    model: xgb.XGBRegressor,
    history: np.ndarray,
    history_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Predict `len(test_dates)` steps ahead from the end of history.

    At each step the current prediction is appended to the buffer so
    that subsequent feature rows can reference it as a lagged value.
    This mirrors the real-world deployment scenario where future
    observations are unavailable.

    Returns
    -------
    preds      : (n_test,) predicted values
    X_test_rec : (n_test, n_features) feature matrix used for each step
    """
    buf        = list(history)
    all_dates  = list(history_dates) + list(test_dates)
    preds      = []
    feat_rows  = []

    for step in range(len(test_dates)):
        t   = len(buf)
        row = _feature_row(buf, t, all_dates[t])
        feat_rows.append(row)
        pred = float(model.predict(row.reshape(1, -1))[0])
        preds.append(pred)
        buf.append(pred)

    X_test_rec = pd.DataFrame(feat_rows, index=test_dates,
                              columns=FEATURE_NAMES)
    return np.array(preds), X_test_rec


# ── SHAP analysis ──────────────────────────────────────────────────────────

def run_shap_analysis(
    models:        dict,          # {series_id: XGBRegressor}
    test_features: dict,          # {series_id: pd.DataFrame shape (40, 17)}
    shap_ids:      list,          # 50-series subsample
    labels:        pd.Series,
    save_dir:      str = SHAP_DIR,
) -> None:
    """
    SHAP analysis on a 50-series subsample.

    Strategy: pool SHAP values across the 50 models (50 × 40 = 2 000 rows)
    to obtain global feature importance estimates.  All models share the
    same feature set so pooling is valid.

    Plots produced
    --------------
    01_beeswarm.png         Global beeswarm summary (all features)
    02_feature_groups.png   Mean |SHAP| per feature group (bar chart)
    03_dependence.png       Dependence plot for the top-ranked feature
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n  Computing SHAP values for {len(shap_ids)} series …")

    all_shap_vals = []
    all_feat_vals = []

    for sid in tqdm(shap_ids, ncols=80, leave=False):
        model   = models[sid]
        X_test  = test_features[sid].values.astype(float)

        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test)   # (40, 17)
        all_shap_vals.append(shap_vals)
        all_feat_vals.append(X_test)

    sv = np.vstack(all_shap_vals)   # (2000, 17)
    fv = np.vstack(all_feat_vals)   # (2000, 17)
    fv_df = pd.DataFrame(fv, columns=FEATURE_NAMES)

    mean_abs = np.abs(sv).mean(axis=0)   # (17,)
    top_feat_idx  = int(np.argmax(mean_abs))
    top_feat_name = FEATURE_NAMES[top_feat_idx]

    # ── Plot 1: beeswarm summary ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv, fv_df,
        feature_names=FEATURE_NAMES,
        plot_type="dot",
        show=False,
        max_display=17,
    )
    plt.title(
        f"Global SHAP beeswarm — XGBoost on F1 dataset\n"
        f"({len(shap_ids)} series × 40 test steps = {len(sv):,} instances)",
        fontsize=11,
    )
    plt.tight_layout()
    _save_fig(plt.gcf(), save_dir, "01_beeswarm.png")

    # ── Plot 2: mean |SHAP| per feature group ───────────────────────────
    group_scores = {}
    for group, feats in FEATURE_GROUPS.items():
        idxs = [FEATURE_NAMES.index(f) for f in feats]
        group_scores[group] = float(np.abs(sv[:, idxs]).mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    groups = list(group_scores.keys())
    scores = [group_scores[g] for g in groups]
    colors = ["#457b9d", "#2a9d8f", "#e9c46a", "#e76f51"]
    bars   = ax.barh(groups, scores, color=colors, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature group importance — mean |SHAP| per group", fontsize=12)
    for bar, v in zip(bars, scores):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    _save_fig(fig, save_dir, "02_feature_groups.png")

    # ── Plot 3: dependence plot for top feature ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        top_feat_idx,
        sv,
        fv_df,
        feature_names=FEATURE_NAMES,
        ax=ax,
        show=False,
    )
    ax.set_title(
        f"SHAP dependence — {top_feat_name}  "
        f"(mean |SHAP| = {mean_abs[top_feat_idx]:.4f})",
        fontsize=11,
    )
    plt.tight_layout()
    _save_fig(fig, save_dir, "03_dependence.png")

    # ── Print feature ranking ────────────────────────────────────────────
    print(f"\n  Top-10 features by mean |SHAP| (pooled across {len(shap_ids)} models):")
    ranking = sorted(zip(FEATURE_NAMES, mean_abs), key=lambda x: -x[1])
    for rank, (name, score) in enumerate(ranking[:10], 1):
        group = next(g for g, fs in FEATURE_GROUPS.items() if name in fs)
        print(f"    {rank:>2}. {name:<14}  {score:.5f}  [{group}]")

    print(f"\n  Top feature: {top_feat_name}")
    print(f"  Group importance: " +
          "  ".join(f"{g}={v:.4f}" for g, v in group_scores.items()))


# ── Summary table ──────────────────────────────────────────────────────────

def _print_summary(results_df: pd.DataFrame) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("XGBOOST RESULTS SUMMARY")
    print(sep)

    overall = results_df.agg({"rmse": ["mean", "median", "std"],
                               "r2":   ["mean", "median"]})
    print(f"\n  {'Metric':<12}  {'Mean':>8}  {'Median':>8}  {'Std':>8}")
    print("  " + "-" * 40)
    print(f"  {'RMSE':<12}  {overall.loc['mean','rmse']:>8.4f}  "
          f"{overall.loc['median','rmse']:>8.4f}  "
          f"{overall.loc['std','rmse']:>8.4f}")
    print(f"  {'R²':<12}  {overall.loc['mean','r2']:>8.4f}  "
          f"{overall.loc['median','r2']:>8.4f}  {'—':>8}")

    print(f"\n  {'Trend label':<14}  {'N':>5}  {'RMSE mean':>10}  {'R² mean':>8}")
    print("  " + "-" * 42)
    for label in ["increasing", "flat", "declining"]:
        sub = results_df[results_df["trend_label"] == label]
        if len(sub):
            print(f"  {label:<14}  {len(sub):>5}  "
                  f"{sub['rmse'].mean():>10.4f}  {sub['r2'].mean():>8.4f}")

    # Comparison against baseline floor
    baseline_rmse = 0.0724   # seasonal naive on same 500-series sample
    xgb_rmse      = results_df["rmse"].mean()
    beat = xgb_rmse < baseline_rmse
    print(f"\n  Baseline floor (seasonal naive, same sample): RMSE = {baseline_rmse:.4f}")
    print(f"  XGBoost mean RMSE                           : RMSE = {xgb_rmse:.4f}")
    print(f"  {'✓ Beats baseline' if beat else '✗ Does NOT beat baseline'}")
    print(sep)


def _save_fig(fig: plt.Figure, directory: str, filename: str) -> None:
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_xgboost(
    parquet_path: str = PARQUET,
    splits_path:  str = SPLITS_JSON,
    results_path: str = RESULTS_CSV,
    n_sample:     int = N_SAMPLE,
    n_shap:       int = N_SHAP,
    seed:         int = SEED,
) -> pd.DataFrame:
    """
    Full XGBoost pipeline:
      1. Load data, compute trend labels, draw stratified 500-series sample
      2. Train one XGBRegressor per series on 221-week history
      3. Predict test split (40 weeks) via recursive forecasting
      4. Evaluate (RMSE, R²) and save results/xgboost_results.csv
      5. SHAP analysis on first `n_shap` series from the sample
    """
    print("=" * 65)
    print("XGBOOST MODEL — F1 Fashion Dataset")
    print("=" * 65)

    # ── Load ─────────────────────────────────────────────────────────────
    print("\nLoading data …")
    history_df, test_df, meta = _load_data()
    n_test = len(test_df)
    print(f"  History  : {len(history_df)} weeks (train + val)")
    print(f"  Test     : {n_test} weeks")
    print(f"  Series   : {history_df.shape[1]:,}")

    # ── Trend labels + sample ─────────────────────────────────────────────
    labels     = _compute_trend_labels(history_df)
    sample_ids = _stratified_sample(labels, n=n_sample, seed=seed)
    lc = labels[sample_ids].value_counts().sort_index()
    print(f"\nSample: {len(sample_ids)} series  "
          f"(declining={lc.get(-1,0)}, flat={lc.get(0,0)}, "
          f"increasing={lc.get(1,0)})")

    label_name = {1: "increasing", 0: "flat", -1: "declining"}

    # ── Train + predict loop ──────────────────────────────────────────────
    print(f"\nTraining {n_sample} XGBoost models …")
    t0       = time.perf_counter()
    rows     = []
    models   = {}
    test_feats = {}

    for sid in tqdm(sample_ids, ncols=80):
        h_vals  = history_df[sid].to_numpy(float)
        h_dates = history_df.index
        t_vals  = test_df[sid].to_numpy(float)
        t_dates = test_df.index

        # Build training features (209 rows after NaN drop)
        X_train, y_train = build_train_features(h_vals, h_dates)

        # Fit model
        model = train_xgb(X_train, y_train)

        # Recursive forecast + collect test feature rows for SHAP
        preds, X_test_rec = recursive_forecast(model, h_vals, h_dates, t_dates)

        # Metrics
        rmse = float(np.sqrt(mean_squared_error(t_vals, preds)))
        r2   = float(r2_score(t_vals, preds))

        m = _series_meta(sid)
        rows.append({
            "series_id":   sid,
            "market":      m["market"],
            "gender":      m["gender"],
            "category":    m["category"],
            "trend_label": label_name[int(labels[sid])],
            "n_train":     len(X_train),
            "rmse":        rmse,
            "r2":          r2,
        })
        models[sid]     = model
        test_feats[sid] = X_test_rec

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/n_sample:.2f}s / series)")

    # ── Save results ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved → {results_path}  ({len(results_df)} rows)")

    # ── Summary ────────────────────────────────────────────────────────────
    _print_summary(results_df)

    # ── SHAP analysis ─────────────────────────────────────────────────────
    # Take every 10th series from the stratified sample to preserve balance
    shap_ids = sample_ids[::10][:n_shap]
    print(f"\n[SHAP] Analysing {len(shap_ids)}-series subsample …")
    run_shap_analysis(models, test_feats, shap_ids, labels)

    return results_df


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_xgboost()
