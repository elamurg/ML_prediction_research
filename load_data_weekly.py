"""
Data Loading Module for Weekly Fashion Trend Forecasting

This module loads and preprocesses:
1. Weekly trend counts (SFS data) - using interpolated columns
2. California weather data - aggregated to weekly
3. Google Trends weekly data

Both XGBoost and LSTM models should use this module for consistent data loading.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def load_weekly_trend_counts(filepath: str) -> pd.DataFrame:
    """
    Load weekly trend counts from CSV.
    
    Uses the INTERPOLATED columns (zara_dress_interp, chanel_bag_interp)
    as they provide smoother signals while preserving trend patterns.
    
    Parameters:
    -----------
    filepath : str
        Path to weekly_trend_counts.csv
    
    Returns:
    --------
    DataFrame with columns:
        - zara_frequency: interpolated zara dress counts
        - chanel_frequency: interpolated chanel bag counts
    """
    df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    
    # Use interpolated columns as target (smoother signal)
    result = pd.DataFrame(index=df.index)
    result['zara_frequency'] = df['zara_dress_interp']
    result['chanel_frequency'] = df['chanel_bag_interp']
    
    # Also keep raw counts for reference if needed
    result['zara_raw'] = df['zara_dress']
    result['chanel_raw'] = df['chanel_bag']
    
    print(f"Loaded weekly trend counts: {len(result)} weeks")
    print(f"  Date range: {result.index.min().strftime('%Y-%m-%d')} to {result.index.max().strftime('%Y-%m-%d')}")
    print(f"  Zara frequency range: {result['zara_frequency'].min():.1f} - {result['zara_frequency'].max():.1f}")
    print(f"  Chanel frequency range: {result['chanel_frequency'].min():.1f} - {result['chanel_frequency'].max():.1f}")
    
    return result


def load_weather_weekly(filepath: str) -> pd.DataFrame:
    """
    Load California weather data and aggregate to weekly.
    
    The raw weather data is hourly, so we aggregate to weekly averages
    to match the trend data frequency.
    
    Parameters:
    -----------
    filepath : str
        Path to California_weather.csv
    
    Returns:
    --------
    DataFrame with weekly weather features indexed by week start date
    """
    # Weather file uses semicolon delimiter
    df = pd.read_csv(filepath, sep=';')
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['dt_iso'].str[:19])
    df = df.set_index('datetime')
    
    # Select relevant weather columns
    weather_cols = ['temp', 'humidity', 'wind_speed', 'clouds_all', 'pressure']
    available_cols = [c for c in weather_cols if c in df.columns]
    
    df_weather = df[available_cols].copy()
    
    # Handle missing values
    df_weather = df_weather.replace('', np.nan)
    for col in df_weather.columns:
        df_weather[col] = pd.to_numeric(df_weather[col], errors='coerce')
    
    # Resample to weekly (starting Sunday to match trend data)
    # 'W-SUN' means week ending on Sunday, so we use 'W-SAT' for week starting Sunday
    weekly = df_weather.resample('W-SUN').agg({
        'temp': 'mean',
        'humidity': 'mean', 
        'wind_speed': 'mean',
        'clouds_all': 'mean',
        'pressure': 'mean'
    })
    
    # Rename columns for clarity
    weekly.columns = [f'weather_{c}' for c in weekly.columns]
    
    # Fill any remaining NaN
    weekly = weekly.ffill().bfill()
    
    print(f"Loaded weather data: {len(weekly)} weeks")
    print(f"  Date range: {weekly.index.min().strftime('%Y-%m-%d')} to {weekly.index.max().strftime('%Y-%m-%d')}")
    
    return weekly


def load_google_trends_weekly(filepath: str) -> pd.DataFrame:
    """
    Load weekly Google Trends data.
    
    Parameters:
    -----------
    filepath : str
        Path to google_trends_weekly_smoothed.csv
    
    Returns:
    --------
    DataFrame with search interest columns indexed by date
    """
    df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    
    print(f"Loaded Google Trends: {len(df)} weeks")
    print(f"  Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    
    return df


def load_all_weekly_data(
    sfs_path: str = 'data/weekly_trend_counts.csv',
    weather_path: str = 'data/California_weather.csv',
    google_trends_path: str = 'data/google_trends_weekly_smoothed.csv'
) -> Dict[str, pd.DataFrame]:
    """
    Load all data sources for weekly forecasting models.
    
    Parameters:
    -----------
    sfs_path : str
        Path to weekly_trend_counts.csv
    weather_path : str
        Path to California_weather.csv
    google_trends_path : str
        Path to google_trends_weekly_smoothed.csv
    
    Returns:
    --------
    Dictionary with keys:
        - 'sfs': Weekly trend counts (interpolated)
        - 'weather': Weekly weather data
        - 'google': Weekly Google Trends data
    """
    print("="*60)
    print("LOADING WEEKLY DATA")
    print("="*60)
    
    # Load each data source
    sfs = load_weekly_trend_counts(sfs_path)
    weather = load_weather_weekly(weather_path)
    google = load_google_trends_weekly(google_trends_path)
    
    # Find common date range
    common_start = max(sfs.index.min(), weather.index.min(), google.index.min())
    common_end = min(sfs.index.max(), weather.index.max(), google.index.max())
    
    print(f"\nCommon date range: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
    
    return {
        'sfs': sfs,
        'weather': weather,
        'google': google
    }


# ============================================================
# MAIN - Test data loading
# ============================================================
if __name__ == "__main__":
    # Test with sample paths
    data = load_all_weekly_data(
        sfs_path='data/weekly_trend_counts.csv',
        weather_path='data/California_weather.csv',
        google_trends_path='data/google_trends_weekly_smoothed.csv'
    )
    
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nSFS shape: {data['sfs'].shape}")
    print(f"Weather shape: {data['weather'].shape}")
    print(f"Google shape: {data['google'].shape}")
    
    print("\nSFS columns:", list(data['sfs'].columns))
    print("Weather columns:", list(data['weather'].columns))
    print("Google columns:", list(data['google'].columns))