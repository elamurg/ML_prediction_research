"""
Load Data - Weekly Aggregation

This module loads and prepares weekly-aggregated fashion trend data
for improved LSTM and TFT performance with larger sample sizes.

Key Changes from Monthly:
- 405 weekly samples vs 94 monthly samples (4x more data)
- 4-week moving average smoothing to reduce noise
- Google Trends included 
- Weather data aggregated to weekly

Note: Google Trends data is only available at monthly granularity.
Interpolating it to weekly would create artificial data points.
For weekly models, we use SFS + Weather + Google Trends.
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
    
    df = pd.read_csv(metadata_path)
    print(f"\nRaw metadata records: {len(df)}")
    
    df['time'] = df['time'].str.replace("Updated on ", "", regex=False)
    df['date'] = pd.to_datetime(df['time'], errors='coerce')
    df['tags'] = df['tags'].fillna("").str.lower()
    
    trend_items = {
        "zara_frequency": ["zara", "dress"],
        "chanel_frequency": ["chanel", "bag"]
    }
    
    for trend_label, keywords in trend_items.items():
        df[trend_label] = df['tags'].apply(
            lambda x: 1 if all(kw in x for kw in keywords) else 0)
   
    print(f"\nTrend matches:")
    print(f"  Zara dress:  {df['zara_frequency'].sum()} posts")
    print(f"  Chanel bag:  {df['chanel_frequency'].sum()} posts")
   
    daily = df.groupby('date')[list(trend_items.keys())].sum()
    daily = daily.asfreq('D', fill_value=0)
    
    weekly = daily.resample('W').sum()
    
    print(f"\nWeekly aggregation:")
    print(f"  Total weeks: {len(weekly)}")
    print(f"  Date range: {weekly.index.min().strftime('%Y-%m-%d')} to {weekly.index.max().strftime('%Y-%m-%d')}")
   
    weekly['zara_frequency'] = weekly['zara_frequency'].rolling(4, min_periods=1).mean()
    weekly['chanel_frequency'] = weekly['chanel_frequency'].rolling(4, min_periods=1).mean()
    
    print(f"\nAfter 4-week smoothing:")
    print(f"  Zara zeros:   {(weekly['zara_frequency'] == 0).sum()}")
    print(f"  Chanel zeros: {(weekly['chanel_frequency'] == 0).sum()}")
    
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
    
    df = pd.read_csv(weather_path, delimiter=';')
    print(f"\nRaw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    if 'dt_iso' in df.columns:
        
        df['datetime'] = pd.to_datetime(df['dt_iso'].str[:19], errors='coerce')
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'dt' in df.columns:
        df['datetime'] = pd.to_datetime(df['dt'], unit='s', errors='coerce')
    else:
        raise ValueError(f"Cannot find date column. Available: {df.columns.tolist()}")
    
    df['date'] = df['datetime'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Date range (hourly): {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total hourly records: {len(df)}")
    
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
    
    agg_dict = {}
    rename_dict = {}
    
    if temp_col:
        agg_dict[temp_col] = 'mean'
        rename_dict[temp_col] = 'avg_temperature'
    
    if humidity_col:
        agg_dict[humidity_col] = 'mean'
        rename_dict[humidity_col] = 'avg_humidity'
    
    if rain_col:
        df[rain_col] = df[rain_col].fillna(0)
        agg_dict[rain_col] = 'sum'
        rename_dict[rain_col] = 'total_rainfall'
    
    if not agg_dict:
        raise ValueError("No weather columns found!")
    
    daily = df.groupby('date').agg(agg_dict)
    weekly = daily.resample('W').mean()
    weekly = weekly.rename(columns=rename_dict)

    if 'avg_temperature' in weekly.columns and weekly['avg_temperature'].mean() > 200:
        weekly['avg_temperature'] = weekly['avg_temperature'] - 273.15
    
    weekly = weekly.ffill().bfill()
    
    print(f"Weekly samples: {len(weekly)}")
    
    return weekly

def load_google_trends_weekly(google_path: str = 'data/google_trends_weekly_smoothed.csv') -> pd.DataFrame:
    """
    Load weekly Google Trends data.
    
    Parameters:
    -----------
    google_path : str
        Path to weekly Google Trends CSV
    
    Returns:
    --------
    DataFrame with weekly search interest
    """
    print("\n" + "=" * 60)
    print("LOADING GOOGLE TRENDS - WEEKLY")
    print("=" * 60)
    
    df = pd.read_csv(google_path, parse_dates=['date'], index_col='date')
    
    print(f"\nGoogle Trends data:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"  Columns: {df.columns.tolist()}")
    
    if 'zara_search_interest' not in df.columns:
        for col in df.columns:
            if 'zara' in col.lower():
                df = df.rename(columns={col: 'zara_search_interest'})
            if 'chanel' in col.lower():
                df = df.rename(columns={col: 'chanel_search_interest'})
    
    return df

def load_all_weekly_data(sfs_metadata_path: str = 'data/SFS_metadata.csv',
                         weather_path: str = 'data/California_weather.csv',
                         google_trends_path: str = 'data/google_trends_weekly_smoothed.csv') -> Dict:
    """
    Load all datasets with weekly aggregation.
    
    Parameters:
    -----------
    sfs_metadata_path : str
        Path to SFS metadata CSV
    weather_path : str
        Path to California weather CSV
    google_trends_path : str
        Path to weekly Google Trends CSV
    
    Returns:
    --------
    Dictionary with 'sfs', 'weather', and 'google' DataFrames
    """
    # Load individual datasets
    sfs_df = load_sfs_weekly(sfs_metadata_path)
    weather_df = load_weather_weekly(weather_path)
    google_df = load_google_trends_weekly(google_trends_path)
    
    # Find common date range across all three datasets
    start_date = max(sfs_df.index.min(), weather_df.index.min(), google_df.index.min())
    end_date = min(sfs_df.index.max(), weather_df.index.max(), google_df.index.max())
    
    print("\n" + "=" * 60)
    print("ALIGNING DATASETS")
    print("=" * 60)
    print(f"\nCommon date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Filter to common range
    sfs_aligned = sfs_df[(sfs_df.index >= start_date) & (sfs_df.index <= end_date)].copy()
    weather_aligned = weather_df[(weather_df.index >= start_date) & (weather_df.index <= end_date)].copy()
    google_aligned = google_df[(google_df.index >= start_date) & (google_df.index <= end_date)].copy()
    
    # Align Google Trends to SFS index (nearest match for any misaligned dates)
    google_aligned = google_aligned.reindex(sfs_aligned.index, method='nearest')
    google_aligned = google_aligned.ffill().bfill()
    
    # Align weather to SFS index
    weather_aligned = weather_aligned.reindex(sfs_aligned.index, method='nearest')
    weather_aligned = weather_aligned.ffill().bfill()
    
    print("\n" + "=" * 60)
    print("WEEKLY DATA LOADING COMPLETE")
    print("=" * 60)
    
    print(f"\nFinal dataset sizes:")
    print(f"  SFS:     {len(sfs_aligned)} weeks, {sfs_aligned.shape[1]} columns")
    print(f"  Weather: {len(weather_aligned)} weeks, {weather_aligned.shape[1]} columns")
    print(f"  Google:  {len(google_aligned)} weeks, {google_aligned.shape[1]} columns")
    
    print(f"\nCompared to monthly (94 samples), you now have {len(sfs_aligned)} samples")
    print(f"That's {len(sfs_aligned) / 94:.1f}x more data for LSTM/TFT!")
    
    return {
        'sfs': sfs_aligned,
        'weather': weather_aligned,
        'google': google_aligned
    }


def find_weekly_peak_date(df: pd.DataFrame, target_column: str) -> pd.Timestamp:
    """
    Find the date when a trend reaches its peak value.
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
        weather_path='data/California_weather.csv',
        google_trends_path='data/google_trends_weekly_smoothed.csv'
    )
    
    # Print sample data
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    
    print("\nSFS Weekly (first 10 rows):")
    print(data['sfs'].head(10))
    
    print("\nWeather Weekly (first 10 rows):")
    print(data['weather'].head(10))
    
    print("\nGoogle Trends Weekly (first 10 rows):")
    print(data['google'].head(10))
    
    # Find peaks
    print("\n" + "=" * 60)
    print("PEAK DETECTION")
    print("=" * 60)
    
    print("\nZara Dress:")
    zara_peak = find_weekly_peak_date(data['sfs'], 'zara_frequency')
    
    print("\nChanel Bag:")
    chanel_peak = find_weekly_peak_date(data['sfs'], 'chanel_frequency')
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # SFS
    axes[0].plot(data['sfs'].index, data['sfs']['zara_frequency'], 
                 color='#2A9D8F', linewidth=2, label='Zara Dress')
    axes[0].plot(data['sfs'].index, data['sfs']['chanel_frequency'], 
                 color='#E76F51', linewidth=2, label='Chanel Bag')
    axes[0].set_title('SFS Weekly Frequency (4-week MA)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Google Trends
    axes[1].plot(data['google'].index, data['google']['zara_search_interest'], 
                 color='#2A9D8F', linewidth=2, label='Zara Dress')
    axes[1].plot(data['google'].index, data['google']['chanel_search_interest'], 
                 color='#E76F51', linewidth=2, label='Chanel Bag')
    axes[1].set_title('Google Trends Weekly Search Interest', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Weather
    axes[2].plot(data['weather'].index, data['weather']['avg_temperature'], 
                 color='#457B9D', linewidth=2)
    axes[2].set_title('Weather - Weekly Avg Temperature', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/weekly_all_data.png', dpi=150)
    print("\nSaved: plots/weekly_all_data.png")
    plt.show()
