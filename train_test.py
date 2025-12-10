"""
Stage 4: Train/Test Split for Time Series
==========================================

This module handles the crucial task of splitting time series data
for training and testing while avoiding data leakage.

Key Concepts Explained:
-----------------------
TIME SERIES SPLITTING is different from regular ML train/test splits!

WRONG WAY (Random Split):
    Randomly shuffle and split data.
    Problem: Model might train on December data and test on January!
    This is DATA LEAKAGE - using future information to predict the past.

RIGHT WAY (Temporal Split):
    Split chronologically - train on past, test on future.
    This simulates real-world forecasting conditions.

YOUR RESEARCH DESIGN:
    Train: Introduction + Growth phases (before peak)
    Test: Saturation + Decline phases (after split point)
    
    This tests if models can predict the decline after only seeing growth!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


def find_peak_date(df: pd.DataFrame, target_column: str) -> pd.Timestamp:
    """
    Find the date when a trend reaches its peak value.
    
    PEAK DETECTION:
    The peak is simply the maximum value in the time series.
    This represents the saturation point - after this, the trend declines.
    
    Parameters:
    -----------
    df : DataFrame
        Data with datetime index
    target_column : str
        Column name of the target variable
    
    Returns:
    --------
    Timestamp of the peak date
    """
    peak_date = df[target_column].idxmax()
    peak_value = df[target_column].max()
    
    print(f"Peak found: {peak_date.strftime('%Y-%m')} with value {peak_value}")
    
    return peak_date


def calculate_split_point(df: pd.DataFrame, 
                          target_column: str, 
                          train_fraction: float = 0.75) -> pd.Timestamp:
    """
    Calculate the train/test split point based on lifecycle position.
    
    SPLIT STRATEGY:
    ---------------
    We want to train on the growth phase and test on saturation/decline.
    
    If train_fraction = 0.75:
    - Training uses the first 75% of data up to the peak
    - Testing uses everything after the split point
    
    Example:
        Data: 100 months total
        Peak at month 80
        Split at month 60 (75% of 80)
        Train: months 1-60
        Test: months 61-100 (includes peak and decline)
    
    WHY 75%?
    - Gives model enough growth data to learn patterns
    - Leaves enough test data to evaluate decline prediction
    - The 70-80% you mentioned is captured in the test set
    
    Parameters:
    -----------
    df : DataFrame
        Data with datetime index
    target_column : str
        Target variable column
    train_fraction : float
        Fraction of pre-peak data to use for training (default 0.75)
    
    Returns:
    --------
    Timestamp for the split point
    """
    # Find peak
    peak_date = find_peak_date(df, target_column)
    peak_position = df.index.get_loc(peak_date)
    
    # Calculate split position (fraction of the way to peak)
    split_position = int(peak_position * train_fraction)
    split_date = df.index[split_position]
    
    # Calculate statistics
    total_months = len(df)
    train_months = split_position + 1
    test_months = total_months - train_months
    
    print(f"\nSplit calculation:")
    print(f"  Total months: {total_months}")
    print(f"  Peak at month: {peak_position + 1} ({100*(peak_position+1)/total_months:.1f}%)")
    print(f"  Split at month: {split_position + 1}")
    print(f"  Training: months 1-{train_months} ({100*train_months/total_months:.1f}%)")
    print(f"  Testing: months {train_months+1}-{total_months} ({100*test_months/total_months:.1f}%)")
    
    return split_date


def create_train_test_split(df: pd.DataFrame,
                           target_column: str,
                           split_date: pd.Timestamp) -> Dict:
    """
    Split data into training and testing sets.
    
    WHAT THIS FUNCTION DOES:
    ------------------------
    1. Splits data at the specified date
    2. Separates features (X) from target (y)
    3. Removes rows with NaN values (from lagged features)
    4. Returns organized dictionary for model training
    
    HANDLING NaN VALUES:
    Lagged features create NaN at the start of the data:
    - lag_1 has NaN for first row
    - lag_12 has NaN for first 12 rows
    
    We drop these rows since they have incomplete information.
    This is safe because we have enough data.
    
    Parameters:
    -----------
    df : DataFrame
        Feature-engineered data
    target_column : str
        Target variable name
    split_date : Timestamp
        Date to split on
    
    Returns:
    --------
    Dictionary with X_train, y_train, X_test, y_test, etc.
    """
    # Split chronologically
    train_df = df[df.index <= split_date].copy()
    test_df = df[df.index > split_date].copy()
    
    # Identify feature columns (everything except target)
    feature_columns = [col for col in df.columns if col != target_column]
    
    # Drop rows with NaN values
    train_df_clean = train_df.dropna()
    test_df_clean = test_df.dropna()
    
    # Separate features and target
    X_train = train_df_clean[feature_columns]
    y_train = train_df_clean[target_column]
    X_test = test_df_clean[feature_columns]
    y_test = test_df_clean[target_column]
    
    # Create result dictionary
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_columns,
        'target_name': target_column,
        'train_dates': train_df_clean.index,
        'test_dates': test_df_clean.index,
        'split_date': split_date,
        'train_size': len(train_df_clean),
        'test_size': len(test_df_clean),
        'n_features': len(feature_columns),
        # Keep full dataframes for reference
        'train_df': train_df_clean,
        'test_df': test_df_clean
    }
    
    return result


def visualize_split(sfs_df: pd.DataFrame,
                   target_column: str,
                   split_date: pd.Timestamp,
                   peak_date: pd.Timestamp,
                   title: str = None,
                   save_path: str = None):
    """
    Visualize the train/test split on the trend lifecycle.
    
    This creates a clear visualization showing:
    - Training data in one color
    - Test data in another color
    - Vertical lines for split point and peak
    
    Parameters:
    -----------
    sfs_df : DataFrame
        Original SFS data
    target_column : str
        Target variable
    split_date : Timestamp
        Where data is split
    peak_date : Timestamp
        Where the peak occurs
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create masks for train/test
    train_mask = sfs_df.index <= split_date
    test_mask = sfs_df.index > split_date
    
    # Plot training data
    ax.plot(sfs_df.index[train_mask], sfs_df.loc[train_mask, target_column],
            color='#2A9D8F', linewidth=2, label='Training Data (Growth Phase)')
    ax.fill_between(sfs_df.index[train_mask], sfs_df.loc[train_mask, target_column],
                    alpha=0.3, color='#2A9D8F')
    
    # Plot test data
    ax.plot(sfs_df.index[test_mask], sfs_df.loc[test_mask, target_column],
            color='#E63946', linewidth=2, label='Test Data (Saturation/Decline)')
    ax.fill_between(sfs_df.index[test_mask], sfs_df.loc[test_mask, target_column],
                    alpha=0.3, color='#E63946')
    
    # Mark split point
    ax.axvline(x=split_date, color='black', linestyle='--', linewidth=2,
               label=f'Split Point: {split_date.strftime("%Y-%m")}')
    
    # Mark peak
    ax.axvline(x=peak_date, color='gold', linestyle=':', linewidth=2,
               label=f'Peak: {peak_date.strftime("%Y-%m")}')
    
    # Formatting
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def prepare_all_splits(feature_datasets: Dict,
                       sfs_df: pd.DataFrame,
                       train_fraction: float = 0.75,
                       visualize: bool = True,
                       save_plots: bool = False) -> Dict:
    """
    Prepare train/test splits for all model configurations.
    
    This creates splits for:
    - Zara: XGBoost, LSTM, TFT
    - Chanel: XGBoost, LSTM, TFT
    
    Each split uses the same split date within a trend (for fair comparison)
    but different trends may have different split dates (different peaks).
    
    Parameters:
    -----------
    feature_datasets : dict
        Dictionary of feature DataFrames from Stage 3
    sfs_df : DataFrame
        Original SFS data for visualization
    train_fraction : float
        Fraction of pre-peak data for training
    visualize : bool
        Whether to show visualizations
    save_plots : bool
        Whether to save plot files
    
    Returns:
    --------
    Dictionary containing all train/test splits
    """
    print("=" * 60)
    print("PREPARING TRAIN/TEST SPLITS")
    print("=" * 60)
    
    splits = {}
    
    # === ZARA DRESS ===
    print("\n" + "-" * 40)
    print("ZARA DRESS")
    print("-" * 40)
    
    # Calculate split point
    zara_split_date = calculate_split_point(
        sfs_df, 'zara_frequency', train_fraction
    )
    zara_peak_date = find_peak_date(sfs_df, 'zara_frequency')
    
    # Create splits for each model
    for model in ['xgb', 'lstm', 'tft']:
        key = f'zara_{model}'
        print(f"\n  Preparing {model.upper()} split...")
        
        splits[key] = create_train_test_split(
            feature_datasets[key],
            'zara_frequency',
            zara_split_date
        )
        
        print(f"    Train: {splits[key]['train_size']} samples")
        print(f"    Test: {splits[key]['test_size']} samples")
        print(f"    Features: {splits[key]['n_features']}")
    
    # Visualize
    if visualize:
        visualize_split(
            sfs_df, 'zara_frequency', zara_split_date, zara_peak_date,
            title='Zara Dress: Train/Test Split',
            save_path='plots/zara_split.png' if save_plots else None
        )
    
    # === CHANEL BAG ===
    print("\n" + "-" * 40)
    print("CHANEL BAG")
    print("-" * 40)
    
    # Calculate split point
    chanel_split_date = calculate_split_point(
        sfs_df, 'chanel_frequency', train_fraction
    )
    chanel_peak_date = find_peak_date(sfs_df, 'chanel_frequency')
    
    # Create splits for each model
    for model in ['xgb', 'lstm', 'tft']:
        key = f'chanel_{model}'
        print(f"\n  Preparing {model.upper()} split...")
        
        splits[key] = create_train_test_split(
            feature_datasets[key],
            'chanel_frequency',
            chanel_split_date
        )
        
        print(f"    Train: {splits[key]['train_size']} samples")
        print(f"    Test: {splits[key]['test_size']} samples")
        print(f"    Features: {splits[key]['n_features']}")
    
    # Visualize
    if visualize:
        visualize_split(
            sfs_df, 'chanel_frequency', chanel_split_date, chanel_peak_date,
            title='Chanel Bag: Train/Test Split',
            save_path='plots/chanel_split.png' if save_plots else None
        )
    
    # Store split dates for reference
    splits['zara_split_date'] = zara_split_date
    splits['zara_peak_date'] = zara_peak_date
    splits['chanel_split_date'] = chanel_split_date
    splits['chanel_peak_date'] = chanel_peak_date
    
    # Summary
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    
    print("\nZara Dress:")
    print(f"  Split date: {zara_split_date.strftime('%Y-%m')}")
    print(f"  Peak date: {zara_peak_date.strftime('%Y-%m')}")
    print(f"  Test includes: {(zara_peak_date - zara_split_date).days // 30} months before peak")
    
    print("\nChanel Bag:")
    print(f"  Split date: {chanel_split_date.strftime('%Y-%m')}")
    print(f"  Peak date: {chanel_peak_date.strftime('%Y-%m')}")
    print(f"  Test includes: {(chanel_peak_date - chanel_split_date).days // 30} months before peak")
    
    return splits


def print_split_info(split_data: Dict):
    """
    Print detailed information about a train/test split.
    
    Parameters:
    -----------
    split_data : dict
        Output from create_train_test_split
    """
    print(f"\n{'='*50}")
    print(f"Target: {split_data['target_name']}")
    print(f"{'='*50}")
    print(f"\nTraining Set:")
    print(f"  Samples: {split_data['train_size']}")
    print(f"  Date range: {split_data['train_dates'].min().strftime('%Y-%m')} to {split_data['train_dates'].max().strftime('%Y-%m')}")
    
    print(f"\nTest Set:")
    print(f"  Samples: {split_data['test_size']}")
    print(f"  Date range: {split_data['test_dates'].min().strftime('%Y-%m')} to {split_data['test_dates'].max().strftime('%Y-%m')}")
    
    print(f"\nFeatures: {split_data['n_features']}")
    print(f"Split date: {split_data['split_date'].strftime('%Y-%m')}")


# ============================================================
# MAIN - Run this file to test splitting
# ============================================================
if __name__ == "__main__":
    import os
    from stage1_data_loading import load_all_data
    from stage3_feature_engineering import prepare_all_model_features
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    data = load_all_data(
        sfs_path='trend_counts_over_time.csv',
        google_path='google_trends.csv',
        weather_path='California_weather.csv'
    )
    
    # Create features
    feature_datasets = prepare_all_model_features(
        data['sfs'],
        data['google'],
        data['weather']
    )
    
    # Create splits
    splits = prepare_all_splits(
        feature_datasets,
        data['sfs'],
        train_fraction=0.75,
        visualize=True,
        save_plots=True
    )
    
    # Show detailed info for one split
    print_split_info(splits['zara_xgb'])