"""
XGBoost Model - Weekly Data Version (SFS Only)

This version uses ONLY the Street Fashion Style data without external 
features for fair comparison with simplified LSTM and TFT models.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class XGBoostForecaster:
    """XGBoost model wrapper."""
    
    def __init__(self, params: Dict = None):
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
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        self.feature_names = list(X_train.columns)
        self.model = xgb.XGBRegressor(**self.params)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        predictions = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        mask = y != 0
        mape = np.mean(np.abs((y[mask] - predictions[mask]) / y[mask])) * 100 if mask.sum() > 0 else np.nan
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'predictions': predictions}
    
    def get_feature_importance(self) -> pd.DataFrame:
        importance = self.model.feature_importances_
        df = pd.DataFrame({'feature': self.feature_names, 'importance': importance})
        return df.sort_values('importance', ascending=False).reset_index(drop=True)


def prepare_sfs_only_features(sfs_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Prepare features using ONLY SFS data."""
    df = pd.DataFrame(index=sfs_df.index)
    df[target_col] = sfs_df[target_col]
    
    # === LAG FEATURES ===
    for lag in [1, 2, 4, 8, 12]:
        df[f'lag_{lag}w'] = df[target_col].shift(lag)
    
    # === ROLLING STATISTICS ===
    for window in [4, 8, 12]:
        df[f'rolling_mean_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).mean()
        df[f'rolling_std_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).std()
    
    # === MOMENTUM ===
    df['momentum_1w'] = df[target_col].diff(1)
    df['momentum_4w'] = df[target_col].diff(4)
    
    # === CALENDAR FEATURES ===
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    df = df.ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def create_train_test_split(features_df: pd.DataFrame, target_col: str, split_at_peak: bool = True) -> Dict:
    """Create train/test split."""
    if split_at_peak:
        peak_idx = features_df[target_col].idxmax()
        peak_loc = features_df.index.get_loc(peak_idx)
        print(f"Peak found at: {peak_idx.strftime('%Y-%m-%d')}, value: {features_df[target_col].max():.2f}")
        train_df = features_df.iloc[:peak_loc + 1]
        test_df = features_df.iloc[peak_loc + 1:]
    else:
        split_point = int(len(features_df) * 0.75)
        train_df = features_df.iloc[:split_point]
        test_df = features_df.iloc[split_point:]
    
    feature_cols = [c for c in features_df.columns if c != target_col]
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    return {
        'X_train': train_df[feature_cols], 'y_train': train_df[target_col],
        'X_test': test_df[feature_cols], 'y_test': test_df[target_col],
        'train_dates': train_df.index, 'test_dates': test_df.index,
    }


def train_and_evaluate_xgboost(split_data: Dict, model_name: str = "XGBoost") -> Dict:
    """Complete XGBoost training and evaluation."""
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*60}")
    
    X_train, y_train = split_data['X_train'], split_data['y_train']
    X_test, y_test = split_data['X_test'], split_data['y_test']
    
    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}, Features: {len(X_train.columns)}")
    
    model = XGBoostForecaster()
    model.train(X_train, y_train, X_test, y_test)
    
    print("\n--- Training Performance ---")
    train_metrics = model.evaluate(X_train, y_train)
    print(f"  RMSE: {train_metrics['rmse']:.2f}, MAE: {train_metrics['mae']:.2f}, R2: {train_metrics['r2']:.3f}")
    
    print("\n--- Test Performance ---")
    test_metrics = model.evaluate(X_test, y_test)
    print(f"  RMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}, R2: {test_metrics['r2']:.3f}, MAPE: {test_metrics['mape']:.1f}%")
    
    return {
        'model': model, 'train_metrics': train_metrics, 'test_metrics': test_metrics,
        'test_predictions': test_metrics['predictions'],
        'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,
        'train_dates': split_data['train_dates'], 'test_dates': split_data['test_dates']
    }


def plot_predictions(results: Dict, title: str = None, save_path: str = None):
    """Visualize predictions."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(results['train_dates'], results['y_train'], color='#2A9D8F', linewidth=1.5, label='Training', alpha=0.8)
    ax.plot(results['test_dates'], results['y_test'], color='#457B9D', linewidth=1.5, label='Test (Actual)')
    ax.plot(results['test_dates'], results['test_predictions'], color='#E63946', linewidth=1.5, 
            linestyle='--', label='Test (Predicted)', alpha=0.8)
    ax.axvline(x=results['train_dates'][-1], color='black', linestyle='--', alpha=0.5)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Frequency')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    metrics = results['test_metrics']
    ax.annotate(f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.2f}", 
                xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import os
    
    os.makedirs('plots', exist_ok=True)
    
    print("\n" + "="*60)
    print("LOADING SFS DATA ONLY")
    print("="*60)
    
    sfs_df = pd.read_csv('data/weekly_trend_counts.csv', parse_dates=['date'], index_col='date')
    sfs_df['zara_frequency'] = sfs_df['zara_dress_interp']
    sfs_df['chanel_frequency'] = sfs_df['chanel_bag_interp']
    print(f"Loaded {len(sfs_df)} weeks of data")
    
    # === ZARA ===
    print("\n" + "="*60)
    print("ZARA DRESS - XGBOOST (SFS ONLY)")
    print("="*60)
    
    zara_features = prepare_sfs_only_features(sfs_df, 'zara_frequency')
    zara_split = create_train_test_split(zara_features, 'zara_frequency')
    zara_results = train_and_evaluate_xgboost(zara_split, "Zara XGBoost (SFS Only)")
    plot_predictions(zara_results, title='Zara Dress: XGBoost (SFS Only)', save_path='plots/zara_xgb_sfs_only.png')
    
    # === CHANEL ===
    print("\n" + "="*60)
    print("CHANEL BAG - XGBOOST (SFS ONLY)")
    print("="*60)
    
    chanel_features = prepare_sfs_only_features(sfs_df, 'chanel_frequency')
    chanel_split = create_train_test_split(chanel_features, 'chanel_frequency')
    chanel_results = train_and_evaluate_xgboost(chanel_split, "Chanel XGBoost (SFS Only)")
    plot_predictions(chanel_results, title='Chanel Bag: XGBoost (SFS Only)', save_path='plots/chanel_xgb_sfs_only.png')
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("XGBOOST (SFS ONLY) SUMMARY")
    print("="*60)
    print(f"\nZara:   R² = {zara_results['test_metrics']['r2']:.3f}, RMSE = {zara_results['test_metrics']['rmse']:.2f}")
    print(f"Chanel: R² = {chanel_results['test_metrics']['r2']:.3f}, RMSE = {chanel_results['test_metrics']['rmse']:.2f}")