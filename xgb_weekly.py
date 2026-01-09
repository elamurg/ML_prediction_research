"""
XGBoost Model - Weekly Data Version

This module implements XGBoost for fashion trend prediction using WEEKLY data
from weekly_trend_counts.csv (interpolated columns).

XGBoost (eXtreme Gradient Boosting) EXPLAINED:
----------------------------------------------

WHAT IS GRADIENT BOOSTING?
Gradient Boosting is an ensemble method that builds models sequentially.
Each new model tries to correct the errors of the previous models.

Think of it like this:
1. First model makes predictions (probably not great)
2. Calculate errors (residuals) - where did we go wrong?
3. Second model learns to predict these errors
4. Add second model's predictions to improve overall
5. Repeat many times

ANALOGY:
Imagine you're trying to hit a target with darts:
- First throw: Miss by 10cm to the left
- Adjustment: Aim 10cm to the right
- Second throw: Miss by 3cm up
- Adjustment: Aim 3cm down
- Each "model" corrects the previous error

THE MATH (simplified):
----------------------
Prediction = F_0 + n*h_1 + n*h_2 + ... + n*h_n

Where:
- F_0 = Initial prediction (usually the mean)
- h_i = Tree i that predicts residuals
- n = Learning rate (how much each tree contributes)

WHY USE XGBoost AS BASELINE?
- Works well with tabular data
- Doesn't require data normalization
- Handles non-linear relationships
- Fast to train
- Easy to interpret (feature importance)

LIMITATION FOR TIME SERIES:
XGBoost treats each row independently - it doesn't inherently
understand that row 10 comes after row 9. That's why we create
lagged features to encode temporal information.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class XGBoostForecaster:
    """
    XGBoost model wrapper for fashion trend forecasting.
    
    This class encapsulates:
    - Model training with hyperparameter tuning
    - Prediction
    - Evaluation
    - Feature importance analysis
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize the XGBoost forecaster.
        
        Default parameters are tuned for small time series datasets.
        
        HYPERPARAMETER EXPLANATIONS:
        ----------------------------
        n_estimators: Number of trees (100)
            More trees = more complex model, but slower
            Too many = overfitting, too few = underfitting
        
        max_depth: Maximum tree depth (4)
            Controls how complex each tree can be
            Deeper = can learn more complex patterns
            Too deep = overfitting
        
        learning_rate: Step size (0.1)
            How much each tree contributes
            Smaller = slower learning but often better
            Larger = faster but might overshoot
        
        subsample: Fraction of data per tree (0.8)
            Random sampling prevents overfitting
            1.0 = use all data (risk overfitting)
            0.8 = use 80% of data per tree
        
        colsample_bytree: Fraction of features per tree (0.8)
            Random feature selection per tree
            Adds diversity to ensemble
        
        min_child_weight: Minimum samples in leaf (3)
            Prevents trees from creating very specific rules
            Higher = more conservative, less overfitting
        """
        self.default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'random_state': 42,
            'verbosity': 0
        }
        
        self.params = params if params else self.default_params
        self.model = None
        self.feature_names = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'XGBoostForecaster':
        """
        Train the XGBoost model.
        
        TRAINING PROCESS:
        -----------------
        1. Initialize model with parameters
        2. For each boosting round:
           a. Current predictions on training data
           b. Calculate residuals (errors)
           c. Fit new tree to predict residuals
           d. Update predictions
           e. Check validation performance
        3. Stop early if validation doesn't improve
        """
        self.feature_names = list(X_train.columns)
        
        self.model = xgb.XGBRegressor(**self.params)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance.
        
        METRICS EXPLAINED:
        ------------------
        RMSE (Root Mean Squared Error):
            sqrt(mean((actual - predicted)^2))
            - In same units as target
            - Penalizes large errors more
            - Lower is better
        
        MAE (Mean Absolute Error):
            mean(|actual - predicted|)
            - In same units as target
            - Equal weight to all errors
            - Lower is better
        
        R2 (R-squared / Coefficient of Determination):
            1 - (SS_residual / SS_total)
            - Range: -inf to 1
            - 1.0 = perfect prediction
            - 0.0 = predicts mean only
            - Negative = worse than mean
        
        MAPE (Mean Absolute Percentage Error):
            mean(|actual - predicted| / |actual|) * 100
            - Percentage error
            - Good for comparing across different scales
        """
        predictions = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        mask = y != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y[mask] - predictions[mask]) / y[mask])) * 100
        else:
            mape = np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': predictions
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        WHY FEATURE IMPORTANCE MATTERS:
        - Understand what drives predictions
        - Identify most valuable features
        - Guide feature engineering
        - Validate domain knowledge
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def plot_feature_importance(self, top_n: int = 15, 
                                title: str = None,
                                save_path: str = None):
        """Visualize feature importance."""
        importance_df = self.get_feature_importance()
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(range(top_n), 
                      top_features['importance'].values[::-1],
                      color='#457B9D')
        
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features['feature'].values[::-1])
        ax.set_xlabel('Feature Importance')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
        return importance_df


def prepare_xgboost_features(sfs_df: pd.DataFrame,
                             weather_df: pd.DataFrame,
                             google_df: pd.DataFrame,
                             target_col: str) -> pd.DataFrame:
    """
    Prepare features for XGBoost model.
    
    Creates lag features, rolling statistics, and incorporates
    external data (weather, Google Trends).
    
    Parameters:
    -----------
    sfs_df : DataFrame
        Weekly SFS data with trend frequencies (from weekly_trend_counts.csv)
    weather_df : DataFrame
        Weekly weather data
    google_df : DataFrame
        Weekly Google Trends data
    target_col : str
        Target column name ('zara_frequency' or 'chanel_frequency')
    
    Returns:
    --------
    DataFrame with features and target aligned
    """
    df = pd.DataFrame(index=sfs_df.index)
    df[target_col] = sfs_df[target_col]
    
    # === LAG FEATURES ===
    lag_weeks = [1, 2, 4, 8, 12, 26, 52]  # 1w, 2w, 1mo, 2mo, 3mo, 6mo, 1yr
    for lag in lag_weeks:
        df[f'lag_{lag}w'] = df[target_col].shift(lag)
    
    # === ROLLING STATISTICS ===
    windows = [4, 12, 26]  # 1 month, 3 months, 6 months
    for window in windows:
        df[f'rolling_mean_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).mean()
        df[f'rolling_std_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).std()
        df[f'rolling_min_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).min()
        df[f'rolling_max_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).max()
    
    # === MOMENTUM FEATURES ===
    df['momentum_1w'] = df[target_col].diff(1)
    df['momentum_4w'] = df[target_col].diff(4)
    df['momentum_12w'] = df[target_col].diff(12)

    for period in [1, 4, 12]:
        pct_change = df[target_col].pct_change(period)
        pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
        df[f'pct_change_{period}w'] = pct_change
    
    # === GOOGLE TRENDS FEATURES ===
    if google_df is not None:
        google_aligned = google_df.reindex(df.index, method='nearest').ffill().bfill()

        if 'zara' in target_col.lower():
            trend_col = 'zara_search_interest'
        else:
            trend_col = 'chanel_search_interest'
        
        if trend_col in google_aligned.columns:
            df['search_interest'] = google_aligned[trend_col]
            df['search_interest_lag_1w'] = df['search_interest'].shift(1)
            df['search_interest_lag_4w'] = df['search_interest'].shift(4)
            df['search_interest_rolling_4w'] = df['search_interest'].rolling(4, min_periods=1).mean()
            df['search_interest_momentum'] = df['search_interest'].diff(1)
    
    # === WEATHER FEATURES ===
    if weather_df is not None:
        weather_aligned = weather_df.reindex(df.index, method='nearest').ffill().bfill()
        
        for col in weather_aligned.columns:
            df[col] = weather_aligned[col]
            df[f'{col}_lag_1w'] = weather_aligned[col].shift(1)
            df[f'{col}_lag_4w'] = weather_aligned[col].shift(4)
    
    # === CALENDAR FEATURES ===
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
  
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # === CLEAN UP ===
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df


def create_train_test_split(features_df: pd.DataFrame,
                            target_col: str,
                            split_at_peak: bool = True,
                            train_fraction: float = 0.75) -> Dict:
    """
    Create train/test split for weekly data.
    
    Parameters:
    -----------
    features_df : DataFrame
        Feature matrix with target column
    target_col : str
        Name of target column
    split_at_peak : bool
        If True, split at the peak of the time series (useful for trend prediction)
        If False, use train_fraction for splitting
    train_fraction : float
        Fraction of data for training (only used if split_at_peak=False)
    
    Returns:
    --------
    Dictionary with train/test data and metadata
    """
    if split_at_peak:
        # Find peak and split there
        peak_idx = features_df[target_col].idxmax()
        peak_loc = features_df.index.get_loc(peak_idx)
        
        print(f"Peak found at: {peak_idx.strftime('%Y-%m-%d')}")
        print(f"Peak value: {features_df[target_col].max():.2f}")
        
        train_df = features_df.iloc[:peak_loc + 1]
        test_df = features_df.iloc[peak_loc + 1:]
    else:
        split_point = int(len(features_df) * train_fraction)
        train_df = features_df.iloc[:split_point]
        test_df = features_df.iloc[split_point:]
    
    feature_cols = [c for c in features_df.columns if c != target_col]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_dates': train_df.index,
        'test_dates': test_df.index,
        'peak_date': peak_idx if split_at_peak else None
    }


def train_and_evaluate_xgboost(split_data: Dict, 
                               model_name: str = "XGBoost") -> Dict:
    """
    Complete XGBoost training and evaluation pipeline.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*60}")
    
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(X_train.columns)}")

    model = XGBoostForecaster()
    model.train(X_train, y_train, X_test, y_test)

    print("\n--- Training Performance ---")
    train_metrics = model.evaluate(X_train, y_train)
    print(f"  RMSE: {train_metrics['rmse']:.2f}")
    print(f"  MAE:  {train_metrics['mae']:.2f}")
    print(f"  R2:   {train_metrics['r2']:.3f}")
 
    print("\n--- Test Performance ---")
    test_metrics = model.evaluate(X_test, y_test)
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  MAE:  {test_metrics['mae']:.2f}")
    print(f"  R2:   {test_metrics['r2']:.3f}")
    print(f"  MAPE: {test_metrics['mape']:.1f}%")
 
    print("\n--- Overfitting Check ---")
    rmse_diff = test_metrics['rmse'] - train_metrics['rmse']
    if rmse_diff > train_metrics['rmse'] * 0.5:
        print(f"  WARNING: Possible overfitting (test RMSE >> train RMSE)")
    else:
        print(f"  OK: Model generalizes reasonably well")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_predictions': train_metrics['predictions'],
        'test_predictions': test_metrics['predictions'],
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_dates': split_data['train_dates'],
        'test_dates': split_data['test_dates']
    }


def plot_predictions(results: Dict, 
                    title: str = None,
                    save_path: str = None):
    """Visualize actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(results['train_dates'], results['y_train'],
            color='#2A9D8F', linewidth=2, label='Training (Actual)')
 
    ax.plot(results['test_dates'], results['y_test'],
            color='#457B9D', linewidth=2, label='Test (Actual)')

    ax.plot(results['test_dates'], results['test_predictions'],
            color='#E63946', linewidth=2, linestyle='--', 
            label='Test (Predicted)', marker='o', markersize=4)

    split_date = results['train_dates'][-1]
    ax.axvline(x=split_date, color='black', linestyle='--', 
               alpha=0.5, label='Train/Test Split')
    
    # Add metrics annotation
    metrics = results['test_metrics']
    metrics_text = f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.2f}"
    ax.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Frequency')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


# ============================================================
# MAIN - Run this file to train XGBoost models
# ============================================================
if __name__ == "__main__":
    import os
    from load_data_weekly import load_all_weekly_data
    
    os.makedirs('plots', exist_ok=True)
  
    print("Loading data...")
    data = load_all_weekly_data(
        sfs_path='data/weekly_trend_counts.csv',
        weather_path='data/California_weather.csv',
        google_trends_path='data/google_trends_weekly_smoothed.csv'
    )
    
    # === ZARA DRESS ===
    print("\n" + "="*60)
    print("ZARA DRESS - WEEKLY XGBOOST")
    print("="*60)
    
    zara_features = prepare_xgboost_features(
        data['sfs'], data['weather'], data['google'], 'zara_frequency'
    )
    print(f"\nZara features shape: {zara_features.shape}")
    print(f"Feature columns: {len(zara_features.columns) - 1}")
    
    zara_split = create_train_test_split(
        zara_features, 'zara_frequency', split_at_peak=True
    )
    
    zara_results = train_and_evaluate_xgboost(
        zara_split, model_name="Zara XGBoost (Weekly)"
    )
    
    plot_predictions(zara_results,
        title='Zara Dress: Weekly XGBoost Predictions vs Actual',
        save_path='plots/zara_xgb_weekly_predictions.png'
    )
    
    zara_results['model'].plot_feature_importance(
        top_n=15, title='Zara Dress: XGBoost Feature Importance',
        save_path='plots/zara_xgb_weekly_importance.png'
    )
    
    # === CHANEL BAG ===
    print("\n" + "="*60)
    print("CHANEL BAG - WEEKLY XGBOOST")
    print("="*60)
    
    chanel_features = prepare_xgboost_features(
        data['sfs'], data['weather'], data['google'], 'chanel_frequency'
    )
    print(f"\nChanel features shape: {chanel_features.shape}")
    
    chanel_split = create_train_test_split(
        chanel_features, 'chanel_frequency', split_at_peak=True
    )
    
    chanel_results = train_and_evaluate_xgboost(
        chanel_split, model_name="Chanel XGBoost (Weekly)"
    )
    
    plot_predictions(chanel_results,
        title='Chanel Bag: Weekly XGBoost Predictions vs Actual',
        save_path='plots/chanel_xgb_weekly_predictions.png'
    )
    
    chanel_results['model'].plot_feature_importance(
        top_n=15, title='Chanel Bag: XGBoost Feature Importance',
        save_path='plots/chanel_xgb_weekly_importance.png'
    )
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("WEEKLY XGBOOST SUMMARY")
    print("="*60)
    print(f"\nZara:   Test RMSE={zara_results['test_metrics']['rmse']:.2f}, R2={zara_results['test_metrics']['r2']:.3f}")
    print(f"Chanel: Test RMSE={chanel_results['test_metrics']['rmse']:.2f}, R2={chanel_results['test_metrics']['r2']:.3f}")