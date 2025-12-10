"""
Stage 3: Feature Engineering
=============================

This module creates features (input variables) for machine learning models.

Key Concepts Explained:
-----------------------
FEATURE ENGINEERING is the process of creating new input variables from
raw data to help machine learning models learn better patterns.

For time series data, we create features that capture:
1. TEMPORAL PATTERNS - How values change over time
2. SEASONALITY - Repeating patterns (monthly, yearly)
3. MOMENTUM - Direction and speed of change
4. EXTERNAL FACTORS - Information from other data sources

WHY IS THIS IMPORTANT?
- Raw data alone may not reveal patterns
- Models need the right inputs to make good predictions
- Good features can dramatically improve model accuracy

CRITICAL CONCEPT - DATA LEAKAGE:
When creating features, we must NEVER use future information.
Each feature must only use data available at or before the prediction time.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def create_lag_features(df: pd.DataFrame, 
                        column: str, 
                        lags: List[int] = [1, 2, 3, 6, 12]) -> pd.DataFrame:
    """
    Create lagged features for a time series column.
    
    LAGGED FEATURES EXPLAINED:
    --------------------------
    A "lag" is a past value of a variable shifted forward to align with
    the current time step.
    
    Example with monthly data:
    
    Original data:
        Month:    Jan   Feb   Mar   Apr   May
        Value:    100   120   110   130   140
    
    After creating lag_1 (1 month ago):
        Month:    Jan   Feb   Mar   Apr   May
        Value:    100   120   110   130   140
        lag_1:    NaN   100   120   110   130   <- shifted forward by 1
    
    After creating lag_2 (2 months ago):
        Month:    Jan   Feb   Mar   Apr   May
        Value:    100   120   110   130   140
        lag_2:    NaN   NaN   100   120   110   <- shifted forward by 2
    
    WHY USE LAGS?
    - lag_1: Captures immediate momentum (what happened last month)
    - lag_3: Captures recent quarter trend
    - lag_6: Captures half-year patterns
    - lag_12: Captures same month last year (seasonality)
    
    Parameters:
    -----------
    df : DataFrame
        Input data with datetime index
    column : str
        Column name to create lags for
    lags : list of int
        Lag periods to create
    
    Returns:
    --------
    DataFrame with original column plus lagged columns
    """
    result = df.copy()
    
    for lag in lags:
        # shift() moves values forward, creating lags
        result[f'{column}_lag_{lag}'] = result[column].shift(lag)
    
    return result


def create_rolling_features(df: pd.DataFrame, 
                           column: str, 
                           windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    ROLLING STATISTICS EXPLAINED:
    -----------------------------
    Rolling statistics compute summary values over a moving window
    of past observations.
    
    Example with window=3 (3-month rolling average):
    
        Month:    Jan   Feb   Mar   Apr   May   Jun
        Value:    100   120   110   130   140   120
        
        Rolling mean calculation:
        - Mar: (100+120+110)/3 = 110
        - Apr: (120+110+130)/3 = 120
        - May: (110+130+140)/3 = 126.7
        - Jun: (130+140+120)/3 = 130
    
    WHAT EACH ROLLING STAT TELLS US:
    
    1. ROLLING MEAN: Smoothed trend
       - Removes noise/volatility
       - Shows underlying direction
       - Example: "On average, the trend has been increasing"
    
    2. ROLLING STD (Standard Deviation): Volatility
       - High std = unstable, lots of variation
       - Low std = stable, consistent pattern
       - Example: "The trend is becoming more volatile"
    
    3. ROLLING MIN/MAX: Range
       - Shows the bounds of recent values
       - Useful for detecting new highs/lows
       - Example: "We're approaching the 3-month high"
    
    Parameters:
    -----------
    df : DataFrame
        Input data with datetime index
    column : str
        Column name for rolling calculations
    windows : list of int
        Window sizes (in months)
    
    Returns:
    --------
    DataFrame with rolling statistics added
    """
    result = df.copy()
    
    for window in windows:
        # Rolling mean - average over window
        result[f'{column}_rolling_mean_{window}'] = \
            result[column].rolling(window=window, min_periods=1).mean()
        
        # Rolling standard deviation - volatility
        result[f'{column}_rolling_std_{window}'] = \
            result[column].rolling(window=window, min_periods=1).std()
        
        # Rolling min and max - range
        result[f'{column}_rolling_min_{window}'] = \
            result[column].rolling(window=window, min_periods=1).min()
        
        result[f'{column}_rolling_max_{window}'] = \
            result[column].rolling(window=window, min_periods=1).max()
    
    return result


def create_momentum_features(df: pd.DataFrame, 
                            column: str, 
                            periods: List[int] = [1, 3, 6]) -> pd.DataFrame:
    """
    Create momentum and rate-of-change features.
    
    MOMENTUM EXPLAINED:
    -------------------
    Momentum measures how much and how fast values are changing.
    
    Two types of momentum features:
    
    1. ABSOLUTE MOMENTUM (difference):
       momentum_3 = current_value - value_3_months_ago
       - Positive = growing
       - Negative = declining
       - Large magnitude = fast change
    
    2. PERCENTAGE CHANGE:
       pct_change_3 = (current - 3_months_ago) / 3_months_ago
       - Normalized measure (independent of scale)
       - 0.10 = 10% increase
       - -0.05 = 5% decrease
    
    WHY MOMENTUM MATTERS FOR FASHION TRENDS:
    - A trend gaining momentum might continue growing
    - Slowing momentum might signal approaching peak
    - Negative momentum indicates decline phase
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    column : str
        Column name
    periods : list of int
        Periods for momentum calculation
    
    Returns:
    --------
    DataFrame with momentum features added
    """
    result = df.copy()
    
    for period in periods:
        # Absolute difference (momentum)
        result[f'{column}_momentum_{period}'] = \
            result[column] - result[column].shift(period)
        
        # Percentage change
        result[f'{column}_pct_change_{period}'] = \
            result[column].pct_change(periods=period)
    
    return result


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features from the datetime index.
    
    CALENDAR FEATURES EXPLAINED:
    ----------------------------
    Time itself contains information! Calendar features capture:
    
    1. MONTH (1-12):
       - Captures monthly seasonality
       - Example: Fashion shows happen in Feb/Mar and Sep/Oct
    
    2. QUARTER (1-4):
       - Captures quarterly patterns
       - Example: Q4 holiday shopping season
    
    3. YEAR:
       - Captures long-term trends
       - Example: Overall growth in fashion interest over years
    
    4. MONTHS_SINCE_START:
       - Simple trend indicator
       - Captures position in the lifecycle
    
    Parameters:
    -----------
    df : DataFrame
        Input data with datetime index
    
    Returns:
    --------
    DataFrame with calendar features added
    """
    result = df.copy()
    
    result['month'] = result.index.month
    result['quarter'] = result.index.quarter
    result['year'] = result.index.year
    result['months_since_start'] = np.arange(len(result))
    
    return result


def create_all_time_series_features(df: pd.DataFrame, 
                                    target_column: str,
                                    lags: List[int] = [1, 2, 3, 6, 12],
                                    rolling_windows: List[int] = [3, 6, 12],
                                    momentum_periods: List[int] = [1, 3, 6]) -> pd.DataFrame:
    """
    Create comprehensive time series features for a target variable.
    
    This combines all feature types:
    - Lagged features
    - Rolling statistics
    - Momentum features
    - Calendar features
    
    Parameters:
    -----------
    df : DataFrame
        Input data with datetime index
    target_column : str
        The main column to create features for
    lags : list
        Lag periods
    rolling_windows : list
        Window sizes for rolling stats
    momentum_periods : list
        Periods for momentum calculation
    
    Returns:
    --------
    DataFrame with all features
    """
    # Start with original data
    result = df[[target_column]].copy()
    
    # Add each feature type
    result = create_lag_features(result, target_column, lags)
    result = create_rolling_features(result, target_column, rolling_windows)
    result = create_momentum_features(result, target_column, momentum_periods)
    result = create_calendar_features(result)
    
    return result


def add_google_trends_features(df: pd.DataFrame, 
                               google_df: pd.DataFrame,
                               trend_column: str,
                               google_column: str,
                               lags: List[int] = [0, 1, 2, 3]) -> pd.DataFrame:
    """
    Add Google Trends data as features.
    
    WHY GOOGLE TRENDS FOR LSTM:
    ---------------------------
    Google search interest can be a LEADING INDICATOR:
    - People search before they buy
    - Search interest might peak before actual adoption
    - Provides external signal beyond just historical frequency
    
    We include multiple lags because:
    - lag_0: Current search interest (contemporaneous)
    - lag_1: What people searched last month
    - lag_2, lag_3: Earlier search patterns
    
    The model can learn which lags are most predictive.
    
    Parameters:
    -----------
    df : DataFrame
        Main feature dataframe
    google_df : DataFrame
        Google Trends data
    trend_column : str
        Name for the trend (e.g., 'zara', 'chanel')
    google_column : str
        Column name in google_df
    lags : list
        Lags to create for Google data
    
    Returns:
    --------
    DataFrame with Google Trends features added
    """
    result = df.copy()
    
    # Join Google data
    result = result.join(google_df[[google_column]], how='left')
    result = result.rename(columns={google_column: f'{trend_column}_search_interest'})
    
    # Create lagged versions
    for lag in lags:
        if lag > 0:  # lag_0 is the original column
            result[f'{trend_column}_search_lag_{lag}'] = \
                result[f'{trend_column}_search_interest'].shift(lag)
    
    return result


def add_weather_features(df: pd.DataFrame, 
                         weather_df: pd.DataFrame,
                         lags: List[int] = [0, 1, 2]) -> pd.DataFrame:
    """
    Add weather data as features.
    
    WHY WEATHER FOR TFT:
    --------------------
    Weather affects fashion choices and might explain seasonality:
    
    1. TEMPERATURE:
       - Hot weather → lighter clothes, dresses
       - Cold weather → coats, heavier items
    
    2. RAINFALL:
       - Rainy periods might affect outdoor photography
       - Seasonal rain patterns in California
    
    3. HUMIDITY:
       - Might affect fabric choices
       - Seasonal comfort considerations
    
    We include lags because weather from previous months might
    influence current fashion choices (e.g., buying clothes for
    expected weather).
    
    Parameters:
    -----------
    df : DataFrame
        Main feature dataframe
    weather_df : DataFrame
        Weather data
    lags : list
        Lags to create for weather data
    
    Returns:
    --------
    DataFrame with weather features added
    """
    result = df.copy()
    
    # Select key weather columns
    weather_cols = ['avg_temperature', 'avg_humidity', 'total_rainfall']
    
    # Join weather data
    result = result.join(weather_df[weather_cols], how='left')
    
    # Create lagged versions
    for col in weather_cols:
        for lag in lags:
            if lag > 0:
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)
    
    return result


def prepare_features_for_xgboost(sfs_df: pd.DataFrame, 
                                 target_column: str) -> pd.DataFrame:
    """
    Prepare features for XGBoost model (SFS only).
    
    XGBoost FEATURE STRATEGY:
    -------------------------
    XGBoost works with tabular data where each row is independent.
    It doesn't inherently understand time, so we create features
    that encode temporal information:
    
    - Lag features: Previous values
    - Rolling stats: Recent trends
    - Momentum: Rate of change
    - Calendar: Time position
    
    This is our BASELINE - using only the SFS frequency data.
    
    Parameters:
    -----------
    sfs_df : DataFrame
        Street Fashion Style data
    target_column : str
        Target variable column name
    
    Returns:
    --------
    DataFrame ready for XGBoost
    """
    print(f"\nPreparing XGBoost features for {target_column}...")
    
    features = create_all_time_series_features(
        sfs_df[[target_column]], 
        target_column,
        lags=[1, 2, 3, 6, 12],
        rolling_windows=[3, 6, 12],
        momentum_periods=[1, 3, 6]
    )
    
    print(f"  Created {len(features.columns)} features")
    print(f"  Features: {list(features.columns)}")
    
    return features


def prepare_features_for_lstm(sfs_df: pd.DataFrame, 
                              google_df: pd.DataFrame,
                              target_column: str,
                              google_column: str) -> pd.DataFrame:
    """
    Prepare features for LSTM model (SFS + Google Trends).
    
    LSTM FEATURE STRATEGY:
    ----------------------
    LSTM processes sequences and learns temporal patterns internally.
    However, we still benefit from:
    
    1. SFS features: Historical frequency patterns
    2. Google Trends: External leading indicator
    
    WHY GOOGLE TRENDS FOR LSTM?
    - Provides additional signal beyond SFS
    - Might capture consumer intent before action
    - LSTM can learn complex relationships between search and adoption
    
    Parameters:
    -----------
    sfs_df : DataFrame
        Street Fashion Style data
    google_df : DataFrame
        Google Trends data
    target_column : str
        Target variable (e.g., 'zara_frequency')
    google_column : str
        Google Trends column (e.g., 'zara_search_interest')
    
    Returns:
    --------
    DataFrame ready for LSTM
    """
    print(f"\nPreparing LSTM features for {target_column}...")
    
    # Start with SFS features
    features = create_all_time_series_features(
        sfs_df[[target_column]], 
        target_column,
        lags=[1, 2, 3, 6, 12],
        rolling_windows=[3, 6, 12],
        momentum_periods=[1, 3, 6]
    )
    
    # Add Google Trends features
    trend_name = target_column.replace('_frequency', '')
    features = add_google_trends_features(
        features, google_df, 
        trend_name, google_column,
        lags=[0, 1, 2, 3]
    )
    
    print(f"  Created {len(features.columns)} features")
    print(f"  Includes Google Trends data")
    
    return features


def prepare_features_for_tft(sfs_df: pd.DataFrame, 
                             google_df: pd.DataFrame,
                             weather_df: pd.DataFrame,
                             target_column: str,
                             google_column: str) -> pd.DataFrame:
    """
    Prepare features for TFT model (SFS + Google Trends + Weather).
    
    TFT FEATURE STRATEGY:
    ---------------------
    Temporal Fusion Transformer can handle multiple input types
    and learn which features are most important through attention.
    
    We give it the richest feature set:
    1. SFS features: Historical frequency patterns
    2. Google Trends: Consumer search behavior
    3. Weather: Seasonality and climate effects
    
    WHY ALL THREE FOR TFT?
    - TFT has attention mechanisms to weight feature importance
    - Can automatically learn which features matter for each time step
    - Weather adds seasonal context that might explain patterns
    - More data allows the model to find complex relationships
    
    Parameters:
    -----------
    sfs_df : DataFrame
        Street Fashion Style data
    google_df : DataFrame
        Google Trends data
    weather_df : DataFrame
        Weather data
    target_column : str
        Target variable
    google_column : str
        Google Trends column
    
    Returns:
    --------
    DataFrame ready for TFT
    """
    print(f"\nPreparing TFT features for {target_column}...")
    
    # Start with SFS features
    features = create_all_time_series_features(
        sfs_df[[target_column]], 
        target_column,
        lags=[1, 2, 3, 6, 12],
        rolling_windows=[3, 6, 12],
        momentum_periods=[1, 3, 6]
    )
    
    # Add Google Trends features
    trend_name = target_column.replace('_frequency', '')
    features = add_google_trends_features(
        features, google_df, 
        trend_name, google_column,
        lags=[0, 1, 2, 3]
    )
    
    # Add Weather features
    features = add_weather_features(
        features, weather_df,
        lags=[0, 1, 2]
    )
    
    print(f"  Created {len(features.columns)} features")
    print(f"  Includes Google Trends + Weather data")
    
    return features


def prepare_all_model_features(sfs_df: pd.DataFrame,
                               google_df: pd.DataFrame,
                               weather_df: pd.DataFrame) -> Dict:
    """
    Prepare feature datasets for all models and both trends.
    
    This creates 6 datasets:
    - Zara: XGBoost, LSTM, TFT
    - Chanel: XGBoost, LSTM, TFT
    
    Parameters:
    -----------
    sfs_df : DataFrame
        Street Fashion Style data
    google_df : DataFrame
        Google Trends data
    weather_df : DataFrame
        Weather data
    
    Returns:
    --------
    Dictionary containing all feature datasets
    """
    print("=" * 60)
    print("PREPARING FEATURES FOR ALL MODELS")
    print("=" * 60)
    
    datasets = {}
    
    # ZARA DRESS
    print("\n--- ZARA DRESS ---")
    datasets['zara_xgb'] = prepare_features_for_xgboost(
        sfs_df, 'zara_frequency'
    )
    datasets['zara_lstm'] = prepare_features_for_lstm(
        sfs_df, google_df, 'zara_frequency', 'zara_search_interest'
    )
    datasets['zara_tft'] = prepare_features_for_tft(
        sfs_df, google_df, weather_df, 'zara_frequency', 'zara_search_interest'
    )
    
    # CHANEL BAG
    print("\n--- CHANEL BAG ---")
    datasets['chanel_xgb'] = prepare_features_for_xgboost(
        sfs_df, 'chanel_frequency'
    )
    datasets['chanel_lstm'] = prepare_features_for_lstm(
        sfs_df, google_df, 'chanel_frequency', 'chanel_search_interest'
    )
    datasets['chanel_tft'] = prepare_features_for_tft(
        sfs_df, google_df, weather_df, 'chanel_frequency', 'chanel_search_interest'
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("FEATURE PREPARATION SUMMARY")
    print("=" * 60)
    for name, df in datasets.items():
        print(f"  {name}: {df.shape[1]} features, {df.shape[0]} samples")
    
    return datasets


# ============================================================
# MAIN - Run this file to test feature engineering
# ============================================================
if __name__ == "__main__":
    from stage1_data_loading import load_all_data
    
    # Load data
    data = load_all_data(
        sfs_path='trend_counts_over_time.csv',
        google_path='google_trends.csv',
        weather_path='California_weather.csv'
    )
    
    # Prepare all features
    feature_datasets = prepare_all_model_features(
        data['sfs'],
        data['google'],
        data['weather']
    )
    
    # Show sample
    print("\n\nSample of Zara XGBoost features:")
    print(feature_datasets['zara_xgb'].head(15))
    
    print("\n\nSample of Zara TFT features (first 10 columns):")
    print(feature_datasets['zara_tft'].iloc[:15, :10])