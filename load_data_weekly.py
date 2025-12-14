"""
Load Data - Weekly Aggregation

This module loads and prepares weekly-aggregated fashion trend data
for improved LSTM and TFT performance with larger sample sizes.

Key Changes from Monthly:
- 405 weekly samples vs 94 monthly samples (4x more data)
- 4-week moving average smoothing to reduce noise
- Google Trends NOT included (only available monthly - no fake interpolation)
- Weather data aggregated to weekly

Note: Google Trends data is only available at monthly granularity.
Interpolating it to weekly would create artificial data points.
For weekly models, we use SFS + Weather only.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_sfs_weekly(metadata_path: str = 'data/SFS_metadata.csv') -> pd.DataFrame:
    """
    Load SFS metadata and aggregate to weekly frequency counts.
    
    Parameters:
    -----------
    metadata_path : str
        Path to SFS_metadata.csv
    
    Returns:
    --------
    DataFrame with weekly trend counts (smoothed)
    """
    print("=" * 60)
    print("LOADING SFS DATA - WEEKLY AGGREGATION")
    print("=" * 60)
    
    # Load raw metadata
    df = pd.read_csv(metadata_path)
    print(f"\nRaw metadata records: {len(df)}")
    
    # Parse dates
    df['time'] = df['time'].str.replace("Updated on ", "", regex=False)
    df['date'] = pd.to_datetime(df['time'], errors='coerce')
    df['tags'] = df['tags'].fillna("").str.lower()
    
    # Define trend items
    trend_items = {
        "zara_frequency": ["zara", "dress"],
        "chanel_frequency": ["chanel", "bag"]
    }
    
    # Tag matching
    for trend_label, keywords in trend_items.items():
        df[trend_label] = df['tags'].apply(
            lambda x: 1 if all(kw in x for kw in keywords) else 0)
    
    # Print match counts
    print(f"\nTrend matches:")
    print(f"  Zara dress:  {df['zara_frequency'].sum()} posts")
    print(f"  Chanel bag:  {df['chanel_frequency'].sum()} posts")
    
    # Aggregate to daily first
    daily = df.groupby('date')[list(trend_items.keys())].sum()
    daily = daily.asfreq('D', fill_value=0)
    
    # Aggregate to weekly
    weekly = daily.resample('W').sum()
    
    print(f"\nWeekly aggregation:")
    print(f"  Total weeks: {len(weekly)}")
    print(f"  Date range: {weekly.index.min().strftime('%Y-%m-%d')} to {weekly.index.max().strftime('%Y-%m-%d')}")
    
    # Apply 4-week moving average smoothing
    weekly['zara_frequency'] = weekly['zara_frequency'].rolling(4, min_periods=1).mean()
    weekly['chanel_frequency'] = weekly['chanel_frequency'].rolling(4, min_periods=1).mean()
    
    print(f"\nAfter 4-week smoothing:")
    print(f"  Zara zeros:   {(weekly['zara_frequency'] == 0).sum()}")
    print(f"  Chanel zeros: {(weekly['chanel_frequency'] == 0).sum()}")
    
    # Ensure datetime index
    weekly.index = pd.to_datetime(weekly.index)
    weekly.index.name = 'date'
    
    return weekly


def load_weather_weekly(weather_path: str = 'data/California_weather.csv') -> pd.DataFrame:
    """
    Load California weather data (hourly) and aggregate to weekly.
    
    The raw data is hourly, so we:
    1. Parse datetime from dt_iso column
    2. Aggregate hourly -> daily (mean for temp/humidity, sum for rain)
    3. Aggregate daily -> weekly
    
    Parameters:
    -----------
    weather_path : str
        Path to California_weather.csv
    
    Returns:
    --------
    DataFrame with weekly weather averages
    """
    print("\n" + "=" * 60)
    print("LOADING WEATHER DATA - WEEKLY AGGREGATION")
    print("=" * 60)
    
    # Weather CSV uses semicolon delimiter
    df = pd.read_csv(weather_path, delimiter=';')
    print(f"\nRaw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Parse datetime - handle different possible formats
    if 'dt_iso' in df.columns:
        # Format: "2008-06-01 00:00:00 +0000 UTC" or similar
        df['datetime'] = pd.to_datetime(df['dt_iso'].str[:19], errors='coerce')
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'dt' in df.columns:
        # Unix timestamp
        df['datetime'] = pd.to_datetime(df['dt'], unit='s', errors='coerce')
    else:
        raise ValueError(f"Cannot find date column. Available: {df.columns.tolist()}")
    
    # Extract date for grouping
    df['date'] = df['datetime'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Date range (hourly): {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total hourly records: {len(df)}")
    
    # Identify weather columns - handle different naming conventions
    temp_col = None
    humidity_col = None
    rain_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'temp' in col_lower and temp_col is None:
            temp_col = col
        if 'humid' in col_lower and humidity_col is None:
            humidity_col = col
        if 'rain' in col_lower and rain_col is None:
            rain_col = col
    
    print(f"\nIdentified columns:")
    print(f"  Temperature: {temp_col}")
    print(f"  Humidity: {humidity_col}")
    print(f"  Rainfall: {rain_col}")
    
    # Build aggregation dict based on available columns
    agg_dict_daily = {}
    agg_dict_weekly = {}
    rename_dict = {}
    
    if temp_col:
        agg_dict_daily[temp_col] = 'mean'
        agg_dict_weekly[temp_col] = 'mean'
        rename_dict[temp_col] = 'avg_temperature'
    
    if humidity_col:
        agg_dict_daily[humidity_col] = 'mean'
        agg_dict_weekly[humidity_col] = 'mean'
        rename_dict[humidity_col] = 'avg_humidity'
    
    if rain_col:
        # Fill NaN rain values with 0 (no rain)
        df[rain_col] = df[rain_col].fillna(0)
        agg_dict_daily[rain_col] = 'sum'
        agg_dict_weekly[rain_col] = 'sum'
        rename_dict[rain_col] = 'total_rainfall'
    
    if not agg_dict_daily:
        raise ValueError("No weather columns found!")
    
    # Step 1: Aggregate hourly -> daily
    daily = df.groupby('date').agg(agg_dict_daily)
    print(f"\nDaily aggregation: {len(daily)} days")
    
    # Step 2: Aggregate daily -> weekly
    weekly = daily.resample('W').agg(agg_dict_weekly)
    print(f"Weekly aggregation: {len(weekly)} weeks")
    
    # Rename columns
    weekly = weekly.rename(columns=rename_dict)
    
    # Convert temperature from Kelvin to Celsius if needed
    if 'avg_temperature' in weekly.columns:
        if weekly['avg_temperature'].mean() > 200:
            weekly['avg_temperature'] = weekly['avg_temperature'] - 273.15
            print("Converted temperature from Kelvin to Celsius")
    
    # Fill any missing values
    weekly = weekly.ffill().bfill()
    
    print(f"\nFinal weekly data:")
    print(f"  Shape: {weekly.shape}")
    print(f"  Date range: {weekly.index.min().strftime('%Y-%m-%d')} to {weekly.index.max().strftime('%Y-%m-%d')}")
    print(f"  Columns: {weekly.columns.tolist()}")
    
    return weekly


def align_weekly_datasets(sfs_df: pd.DataFrame, 
                          weather_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align SFS and Weather datasets to a common weekly date range.
    
    Note: Google Trends is excluded as it's only available monthly.
    
    Parameters:
    -----------
    sfs_df : DataFrame
        Weekly SFS data
    weather_df : DataFrame
        Weekly weather data
    
    Returns:
    --------
    Tuple of aligned DataFrames (sfs, weather)
    """
    print("\n" + "=" * 60)
    print("ALIGNING DATASETS TO COMMON WEEKLY RANGE")
    print("=" * 60)
    
    # Find common date range
    start_date = max(sfs_df.index.min(), weather_df.index.min())
    end_date = min(sfs_df.index.max(), weather_df.index.max())
    
    print(f"\nIndividual date ranges:")
    print(f"  SFS:     {sfs_df.index.min().strftime('%Y-%m-%d')} to {sfs_df.index.max().strftime('%Y-%m-%d')}")
    print(f"  Weather: {weather_df.index.min().strftime('%Y-%m-%d')} to {weather_df.index.max().strftime('%Y-%m-%d')}")
    
    print(f"\nCommon range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Filter to common range
    sfs_aligned = sfs_df[(sfs_df.index >= start_date) & (sfs_df.index <= end_date)].copy()
    weather_aligned = weather_df[(weather_df.index >= start_date) & (weather_df.index <= end_date)].copy()
    
    # Ensure same index
    common_index = sfs_aligned.index.intersection(weather_aligned.index)
    
    sfs_aligned = sfs_aligned.loc[common_index]
    weather_aligned = weather_aligned.loc[common_index]
    
    print(f"\nAligned dataset sizes:")
    print(f"  SFS:     {len(sfs_aligned)} weeks")
    print(f"  Weather: {len(weather_aligned)} weeks")
    
    return sfs_aligned, weather_aligned


def load_all_weekly_data(sfs_metadata_path: str = 'data/SFS_metadata.csv',
                         weather_path: str = 'data/California_weather.csv') -> Dict:
    """
    Load all datasets with weekly aggregation.
    
    This is the main entry point for the weekly data pipeline.
    
    Note: Google Trends is NOT included because it's only available monthly.
    Interpolating monthly to weekly would create artificial data points.
    
    Parameters:
    -----------
    sfs_metadata_path : str
        Path to SFS metadata CSV
    weather_path : str
        Path to California weather CSV
    
    Returns:
    --------
    Dictionary with 'sfs' and 'weather' DataFrames
    (No 'google' key - not available at weekly granularity)
    """
    # Load individual datasets
    sfs_df = load_sfs_weekly(sfs_metadata_path)
    weather_df = load_weather_weekly(weather_path)
    
    # Align to common range
    sfs_aligned, weather_aligned = align_weekly_datasets(sfs_df, weather_df)
    
    print("\n" + "=" * 60)
    print("WEEKLY DATA LOADING COMPLETE")
    print("=" * 60)
    
    print(f"\nFinal dataset sizes:")
    print(f"  SFS:     {len(sfs_aligned)} weeks, {sfs_aligned.shape[1]} columns")
    print(f"  Weather: {len(weather_aligned)} weeks, {weather_aligned.shape[1]} columns")
    
    print(f"\nCompared to monthly (94 samples), you now have {len(sfs_aligned)} samples")
    print(f"That's {len(sfs_aligned) / 94:.1f}x more data for LSTM/TFT!")
    
    print(f"\nNote: Google Trends NOT included (only available monthly)")
    
    return {
        'sfs': sfs_aligned,
        'weather': weather_aligned
        # No 'google' key - not available at weekly granularity
    }


def find_weekly_peak_date(df: pd.DataFrame, target_column: str) -> pd.Timestamp:
    """
    Find the date when a trend reaches its peak value.
    
    Parameters:
    -----------
    df : DataFrame
        Weekly data with datetime index
    target_column : str
        Column name of the target variable
    
    Returns:
    --------
    Timestamp of the peak date
    """
    peak_date = df[target_column].idxmax()
    peak_value = df[target_column].max()
    
    print(f"Peak found: {peak_date.strftime('%Y-%m-%d')} with value {peak_value:.2f}")
    
    return peak_date


# ============================================================
# MAIN - Run this file to test weekly data loading
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Load all weekly data
    data = load_all_weekly_data(
        sfs_metadata_path='data/SFS_metadata.csv',
        weather_path='data/California_weather.csv'
    )
    
    # Print sample data
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    
    print("\nSFS Weekly (first 10 rows):")
    print(data['sfs'].head(10))
    
    print("\nWeather Weekly (first 10 rows):")
    print(data['weather'].head(10))
    
    # Find peaks
    print("\n" + "=" * 60)
    print("PEAK DETECTION")
    print("=" * 60)
    
    print("\nZara Dress:")
    zara_peak = find_weekly_peak_date(data['sfs'], 'zara_frequency')
    
    print("\nChanel Bag:")
    chanel_peak = find_weekly_peak_date(data['sfs'], 'chanel_frequency')
    
    # Calculate train/test split sizes
    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT PREVIEW (at peak)")
    print("=" * 60)
    
    zara_peak_idx = data['sfs'].index.get_loc(zara_peak)
    chanel_peak_idx = data['sfs'].index.get_loc(chanel_peak)
    
    print(f"\nZara (split at peak):")
    print(f"  Training: {zara_peak_idx + 1} weeks")
    print(f"  Testing:  {len(data['sfs']) - zara_peak_idx - 1} weeks")
    
    print(f"\nChanel (split at peak):")
    print(f"  Training: {chanel_peak_idx + 1} weeks")
    print(f"  Testing:  {len(data['sfs']) - chanel_peak_idx - 1} weeks")
    
    # Compare to monthly
    print("\n" + "=" * 60)
    print("COMPARISON: WEEKLY vs MONTHLY SAMPLE SIZES")
    print("=" * 60)
    print(f"\n{'':20} {'Monthly':>12} {'Weekly':>12} {'Improvement':>12}")
    print("-" * 60)
    print(f"{'Zara Train':20} {'39':>12} {zara_peak_idx + 1:>12} {(zara_peak_idx + 1) / 39:.1f}x")
    print(f"{'Zara Test':20} {'43':>12} {len(data['sfs']) - zara_peak_idx - 1:>12} {(len(data['sfs']) - zara_peak_idx - 1) / 43:.1f}x")
    print(f"{'Chanel Train':20} {'49':>12} {chanel_peak_idx + 1:>12} {(chanel_peak_idx + 1) / 49:.1f}x")
    print(f"{'Chanel Test':20} {'19':>12} {len(data['sfs']) - chanel_peak_idx - 1:>12} {(len(data['sfs']) - chanel_peak_idx - 1) / 19:.1f}x")
    
    # Plot the weekly data
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Zara
    axes[0].plot(data['sfs'].index, data['sfs']['zara_frequency'], 
                 color='#2A9D8F', linewidth=2, label='Zara Dress')
    axes[0].axvline(x=zara_peak, color='red', linestyle='--', label=f'Peak: {zara_peak.strftime("%Y-%m-%d")}')
    axes[0].set_title('Zara Dress - Weekly Frequency (4-week MA)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Chanel
    axes[1].plot(data['sfs'].index, data['sfs']['chanel_frequency'], 
                 color='#E76F51', linewidth=2, label='Chanel Bag')
    axes[1].axvline(x=chanel_peak, color='red', linestyle='--', label=f'Peak: {chanel_peak.strftime("%Y-%m-%d")}')
    axes[1].set_title('Chanel Bag - Weekly Frequency (4-week MA)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/weekly_trends_with_peaks.png', dpi=150)
    print("\nSaved: plots/weekly_trends_with_peaks.png")
    plt.show()
    