"""
Stage 1: Data Loading and Cleaning
===================================

This module handles loading and cleaning all three datasets:
1. Street Fashion Style (SFS) - trend frequency counts
2. Google Trends - search interest data
3. California Weather - seasonal weather patterns

Key Concepts Explained:
-----------------------
DATA CLEANING is the process of preparing raw data for analysis by:
- Handling missing values
- Converting data types (especially dates)
- Standardizing column names
- Aligning datasets to common formats

Why is this important?
- Machine learning models require clean, consistent data
- Mismatched dates or formats will cause errors
- Missing values can bias model predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_sfs_data(filepath):
    """
    Load and clean the Street Fashion Style dataset.
    
    This dataset contains monthly counts of fashion items observed
    in street photography. It's our TARGET VARIABLE - what we're
    trying to predict.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    DataFrame with datetime index and cleaned column names
    """
    # Load the data
    df = pd.read_csv(filepath)
    
    print("=" * 60)
    print("LOADING STREET FASHION STYLE (SFS) DATA")
    print("=" * 60)
    print(f"\nRaw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create a clean copy
    df_clean = df.copy()
    
    # Parse the Month column to datetime
    # pd.to_datetime() converts strings like "2008-03" to datetime objects
    df_clean['date'] = pd.to_datetime(df_clean['Month'])
    
    # Rename columns to be consistent (lowercase, underscores instead of spaces)
    # This makes the code easier to work with
    df_clean = df_clean.rename(columns={
        'zara dress': 'zara_frequency',
        'chanel bag': 'chanel_frequency'
    })
    
    # Set date as the index
    # Using datetime as index enables time-series operations
    df_clean = df_clean.set_index('date')
    df_clean = df_clean.drop(columns=['Month'])
    
    print(f"\nCleaned data shape: {df_clean.shape}")
    print(f"Date range: {df_clean.index.min().strftime('%Y-%m')} to {df_clean.index.max().strftime('%Y-%m')}")
    print(f"Total months: {len(df_clean)}")
    
    # Check for any issues
    print(f"\nMissing values: {df_clean.isna().sum().sum()}")
    print(f"Zero values in Zara: {(df_clean['zara_frequency'] == 0).sum()}")
    print(f"Zero values in Chanel: {(df_clean['chanel_frequency'] == 0).sum()}")
    
    return df_clean


def load_google_trends_data(filepath):
    """
    Load and clean the Google Trends dataset.
    
    Google Trends provides normalized search interest (0-100 scale) for
    search terms over time. This data often LEADS actual behavior -
    people search for things before they buy/wear them.
    
    HYPOTHESIS: Search interest might be a leading indicator of
    fashion trend adoption.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file (semicolon-delimited)
    
    Returns:
    --------
    DataFrame with datetime index and cleaned column names
    """
    # Load the data - note semicolon delimiter
    df = pd.read_csv(filepath, delimiter=';')
    
    print("\n" + "=" * 60)
    print("LOADING GOOGLE TRENDS DATA")
    print("=" * 60)
    print(f"\nRaw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    df_clean = df.copy()
    
    # Parse dates
    df_clean['date'] = pd.to_datetime(df_clean['Month'])
    
    # Rename columns
    df_clean = df_clean.rename(columns={
        'Zara Dress': 'zara_search_interest',
        'Chanel Bag': 'chanel_search_interest'
    })
    
    # Set date as index
    df_clean = df_clean.set_index('date')
    df_clean = df_clean.drop(columns=['Month'])
    
    print(f"\nCleaned data shape: {df_clean.shape}")
    print(f"Date range: {df_clean.index.min().strftime('%Y-%m')} to {df_clean.index.max().strftime('%Y-%m')}")
    print(f"Total months: {len(df_clean)}")
    print(f"\nSearch interest range: {df_clean.min().min()} to {df_clean.max().max()}")
    
    return df_clean


def load_weather_data(filepath):
    """
    Load and clean the California weather dataset.
    
    This dataset contains HOURLY weather observations. Since our other
    data is MONTHLY, we need to AGGREGATE (summarize) the hourly data
    to monthly averages/totals.
    
    WHY WEATHER DATA?
    Fashion is seasonal - people wear different things based on weather.
    Temperature, rain, and humidity might influence fashion choices.
    
    AGGREGATION EXPLAINED:
    - Hourly data: 24 readings per day × ~30 days = ~720 readings/month
    - We summarize these into single monthly values:
      - Temperature: monthly AVERAGE (mean)
      - Rainfall: monthly TOTAL (sum)
      - Humidity: monthly AVERAGE (mean)
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file (semicolon-delimited)
    
    Returns:
    --------
    DataFrame with monthly aggregated weather data
    """
    # Load the data
    df = pd.read_csv(filepath, delimiter=';')
    
    print("\n" + "=" * 60)
    print("LOADING CALIFORNIA WEATHER DATA")
    print("=" * 60)
    print(f"\nRaw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    df_clean = df.copy()
    
    # Parse the datetime column
    # The dt_iso column format: "2008-01-01 00:00:00 +0000 UTC"
    # We take the first 19 characters to get "2008-01-01 00:00:00"
    df_clean['datetime'] = pd.to_datetime(df_clean['dt_iso'].str[:19])
    
    # Extract year-month for aggregation
    # pd.Period creates a period object (like "2008-01") for grouping
    df_clean['year_month'] = df_clean['datetime'].dt.to_period('M')
    
    print(f"\nHourly readings: {len(df_clean)}")
    print(f"Date range: {df_clean['datetime'].min()} to {df_clean['datetime'].max()}")
    
    # AGGREGATE TO MONTHLY
    # groupby('year_month') groups all rows from the same month together
    # agg() applies different functions to different columns
    weather_monthly = df_clean.groupby('year_month').agg({
        'temp': 'mean',           # Average temperature
        'humidity': 'mean',       # Average humidity
        'wind_speed': 'mean',     # Average wind speed
        'clouds_all': 'mean',     # Average cloud cover
        'rain_1h': 'sum',         # TOTAL rainfall (sum, not average!)
    }).reset_index()
    
    # Convert period back to datetime for consistent indexing
    weather_monthly['date'] = weather_monthly['year_month'].dt.to_timestamp()
    weather_monthly = weather_monthly.set_index('date')
    weather_monthly = weather_monthly.drop(columns=['year_month'])
    
    # Rename columns to be more descriptive
    weather_monthly = weather_monthly.rename(columns={
        'temp': 'avg_temperature',
        'humidity': 'avg_humidity',
        'wind_speed': 'avg_wind_speed',
        'clouds_all': 'avg_cloud_cover',
        'rain_1h': 'total_rainfall'
    })
    
    print(f"\nAggregated to monthly: {weather_monthly.shape}")
    print(f"Monthly date range: {weather_monthly.index.min().strftime('%Y-%m')} to {weather_monthly.index.max().strftime('%Y-%m')}")
    
    # Handle any missing values by forward filling
    # Forward fill: use the previous valid value
    missing_before = weather_monthly.isna().sum().sum()
    weather_monthly = weather_monthly.ffill().bfill()
    missing_after = weather_monthly.isna().sum().sum()
    
    if missing_before > 0:
        print(f"\nMissing values filled: {missing_before} -> {missing_after}")
    
    return weather_monthly


def align_datasets(sfs_df, google_df, weather_df):
    """
    Align all datasets to a common date range.
    
    WHY ALIGNMENT IS NECESSARY:
    - Different datasets may cover different time periods
    - We can only use dates where ALL datasets have data
    - Misaligned data would cause errors or missing values
    
    Parameters:
    -----------
    sfs_df : DataFrame - Street Fashion Style data
    google_df : DataFrame - Google Trends data  
    weather_df : DataFrame - Weather data
    
    Returns:
    --------
    Tuple of (sfs_aligned, google_aligned, weather_aligned, date_info)
    """
    print("\n" + "=" * 60)
    print("ALIGNING DATASETS TO COMMON DATE RANGE")
    print("=" * 60)
    
    # Find the overlapping date range
    # We need the LATEST start date and EARLIEST end date
    common_start = max(
        sfs_df.index.min(), 
        google_df.index.min(), 
        weather_df.index.min()
    )
    common_end = min(
        sfs_df.index.max(), 
        google_df.index.max(), 
        weather_df.index.max()
    )
    
    print(f"\nIndividual date ranges:")
    print(f"  SFS:     {sfs_df.index.min().strftime('%Y-%m')} to {sfs_df.index.max().strftime('%Y-%m')}")
    print(f"  Google:  {google_df.index.min().strftime('%Y-%m')} to {google_df.index.max().strftime('%Y-%m')}")
    print(f"  Weather: {weather_df.index.min().strftime('%Y-%m')} to {weather_df.index.max().strftime('%Y-%m')}")
    print(f"\nCommon range: {common_start.strftime('%Y-%m')} to {common_end.strftime('%Y-%m')}")
    
    # Filter each dataset to the common range
    sfs_aligned = sfs_df[(sfs_df.index >= common_start) & (sfs_df.index <= common_end)].copy()
    google_aligned = google_df[(google_df.index >= common_start) & (google_df.index <= common_end)].copy()
    weather_aligned = weather_df[(weather_df.index >= common_start) & (weather_df.index <= common_end)].copy()
    
    print(f"\nAligned dataset sizes:")
    print(f"  SFS:     {len(sfs_aligned)} months")
    print(f"  Google:  {len(google_aligned)} months")
    print(f"  Weather: {len(weather_aligned)} months")
    
    # Store date info for later use
    date_info = {
        'common_start': common_start,
        'common_end': common_end,
        'n_months': len(sfs_aligned)
    }
    
    return sfs_aligned, google_aligned, weather_aligned, date_info


def load_all_data(sfs_path, google_path, weather_path):
    """
    Main function to load and prepare all datasets.
    
    This is a convenience function that runs the entire data loading
    pipeline and returns everything needed for the next stage.
    
    Parameters:
    -----------
    sfs_path : str - Path to SFS CSV
    google_path : str - Path to Google Trends CSV
    weather_path : str - Path to Weather CSV
    
    Returns:
    --------
    Dictionary containing all cleaned and aligned datasets
    """
    # Load each dataset
    sfs_df = load_sfs_data(sfs_path)
    google_df = load_google_trends_data(google_path)
    weather_df = load_weather_data(weather_path)
    
    # Align to common date range
    sfs_aligned, google_aligned, weather_aligned, date_info = align_datasets(
        sfs_df, google_df, weather_df
    )
    
    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE")
    print("=" * 60)
    
    return {
        'sfs': sfs_aligned,
        'google': google_aligned,
        'weather': weather_aligned,
        'date_info': date_info,
        # Also keep full datasets for reference
        'sfs_full': sfs_df,
        'google_full': google_df,
        'weather_full': weather_df
    }


# ============================================================
# MAIN - Run this file directly to test data loading
# ============================================================
if __name__ == "__main__":
    # Example usage - update paths as needed
    data = load_all_data(
        sfs_path='trend_counts_over_time.csv',
        google_path='google_trends.csv',
        weather_path='California_weather.csv'
    )
    
    print("\n\nSample of each dataset:")
    print("\n--- SFS Data ---")
    print(data['sfs'].head())
    
    print("\n--- Google Trends Data ---")
    print(data['google'].head())
    
    print("\n--- Weather Data ---")
    print(data['weather'].head())