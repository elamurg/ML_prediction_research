"""
Exploratory Data Analysis (EDA)

This module performs visual and statistical exploration of the data
to understand patterns, relationships, and characteristics.

Key Concepts Explained:
-----------------------
EDA is the process of investigating data to:
1. Understand the structure and distribution of variables
2. Identify patterns, trends, and anomalies
3. Find relationships between variables
4. Inform feature engineering and modeling decisions

Why EDA matters for your research:
- Helps identify where peaks (saturation points) occur
- Shows if Google Trends leads or lags actual adoption
- Reveals seasonal patterns in weather data
- Validates assumptions about trend lifecycles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_all_data

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def plot_trend_lifecycles(sfs_df, save_path=None):
    """
    Visualize the complete lifecycle of both fashion trends.
    
    TREND LIFECYCLE EXPLAINED:
    Fashion trends follow a pattern similar to product adoption:
    1. Introduction: Few early adopters, low frequency
    2. Growth: Rapid increase as trend catches on
    3. Peak/Saturation: Maximum popularity reached
    4. Decline: Trend fades, frequency decreases
    
    This visualization helps us:
    - See the overall shape of each trend
    - Identify where the peak occurs
    - Understand the difference between the two trends
    
    Parameters:
    -----------
    sfs_df : DataFrame
        Street Fashion Style data with datetime index
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # === ZARA DRESS ===
    ax1 = axes[0]
    ax1.plot(sfs_df.index, sfs_df['zara_frequency'], 
             color='#E63946', linewidth=2, marker='o', markersize=4, 
             label='Zara Dress Frequency')
    ax1.fill_between(sfs_df.index, sfs_df['zara_frequency'], 
                     alpha=0.3, color='#E63946')
    
    # Find and mark the peak
    zara_peak_idx = sfs_df['zara_frequency'].idxmax()
    zara_peak_val = sfs_df['zara_frequency'].max()
    ax1.axvline(x=zara_peak_idx, color='darkred', linestyle='--', 
                alpha=0.7, label=f'Peak: {zara_peak_idx.strftime("%Y-%m")}')
    ax1.scatter([zara_peak_idx], [zara_peak_val], color='darkred', 
                s=150, zorder=5, marker='*')
    
    ax1.set_title('Zara Dress Trend Lifecycle', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Frequency Count')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # === CHANEL BAG ===
    ax2 = axes[1]
    ax2.plot(sfs_df.index, sfs_df['chanel_frequency'], 
             color='#457B9D', linewidth=2, marker='o', markersize=4,
             label='Chanel Bag Frequency')
    ax2.fill_between(sfs_df.index, sfs_df['chanel_frequency'], 
                     alpha=0.3, color='#457B9D')
    
    # Find and mark the peak
    chanel_peak_idx = sfs_df['chanel_frequency'].idxmax()
    chanel_peak_val = sfs_df['chanel_frequency'].max()
    ax2.axvline(x=chanel_peak_idx, color='darkblue', linestyle='--', 
                alpha=0.7, label=f'Peak: {chanel_peak_idx.strftime("%Y-%m")}')
    ax2.scatter([chanel_peak_idx], [chanel_peak_val], color='darkblue', 
                s=150, zorder=5, marker='*')
    
    ax2.set_title('Chanel Bag Trend Lifecycle', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Frequency Count')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print peak analysis
    print("\n" + "=" * 50)
    print("PEAK ANALYSIS")
    print("=" * 50)
    
    zara_peak_pos = sfs_df.index.get_loc(zara_peak_idx) + 1
    chanel_peak_pos = sfs_df.index.get_loc(chanel_peak_idx) + 1
    
    print(f"\nZara Dress:")
    print(f"  Peak date: {zara_peak_idx.strftime('%Y-%m')}")
    print(f"  Peak value: {zara_peak_val}")
    print(f"  Position: month {zara_peak_pos} of {len(sfs_df)} ({100*zara_peak_pos/len(sfs_df):.1f}%)")
    
    print(f"\nChanel Bag:")
    print(f"  Peak date: {chanel_peak_idx.strftime('%Y-%m')}")
    print(f"  Peak value: {chanel_peak_val}")
    print(f"  Position: month {chanel_peak_pos} of {len(sfs_df)} ({100*chanel_peak_pos/len(sfs_df):.1f}%)")
    
    return {
        'zara_peak_date': zara_peak_idx,
        'zara_peak_value': zara_peak_val,
        'chanel_peak_date': chanel_peak_idx,
        'chanel_peak_value': chanel_peak_val
    }


def plot_sfs_vs_google_trends(sfs_df, google_df, save_path=None):
    """
    Compare Street Fashion Style with Google Trends data.
    
    KEY HYPOTHESIS:
    Google search interest might be a LEADING INDICATOR of fashion trends.
    People search for things before they buy/wear them.
    
    If this is true, we should see:
    - Google Trends peaks BEFORE or AT THE SAME TIME as SFS
    - High correlation between search interest and frequency
    
    This visualization uses DUAL Y-AXES because the scales are different:
    - SFS: Raw frequency counts (varies widely)
    - Google: Normalized interest 0-100
    
    Parameters:
    -----------
    sfs_df : DataFrame - Street Fashion Style data
    google_df : DataFrame - Google Trends data
    save_path : str, optional - Path to save figure
    """
    # Ensure we're using overlapping dates
    common_dates = sfs_df.index.intersection(google_df.index)
    sfs_aligned = sfs_df.loc[common_dates]
    google_aligned = google_df.loc[common_dates]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # === ZARA DRESS ===
    ax1 = axes[0]
    ax1_twin = ax1.twinx()  # Create second y-axis
    
    # Plot SFS on primary axis
    line1, = ax1.plot(sfs_aligned.index, sfs_aligned['zara_frequency'], 
                      color='#E63946', linewidth=2, label='SFS Frequency')
    # Plot Google on secondary axis
    line2, = ax1_twin.plot(google_aligned.index, google_aligned['zara_search_interest'], 
                           color='#2A9D8F', linewidth=2, linestyle='--', 
                           label='Google Search Interest')
    
    ax1.set_ylabel('SFS Frequency', color='#E63946')
    ax1_twin.set_ylabel('Google Search Interest (0-100)', color='#2A9D8F')
    ax1.set_title('Zara Dress: Street Fashion vs Google Search', 
                  fontsize=14, fontweight='bold')
    ax1.legend(handles=[line1, line2], loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # === CHANEL BAG ===
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    line3, = ax2.plot(sfs_aligned.index, sfs_aligned['chanel_frequency'], 
                      color='#457B9D', linewidth=2, label='SFS Frequency')
    line4, = ax2_twin.plot(google_aligned.index, google_aligned['chanel_search_interest'], 
                           color='#F4A261', linewidth=2, linestyle='--',
                           label='Google Search Interest')
    
    ax2.set_ylabel('SFS Frequency', color='#457B9D')
    ax2_twin.set_ylabel('Google Search Interest (0-100)', color='#F4A261')
    ax2.set_title('Chanel Bag: Street Fashion vs Google Search', 
                  fontsize=14, fontweight='bold')
    ax2.legend(handles=[line3, line4], loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return sfs_aligned, google_aligned


def analyze_correlations(sfs_df, google_df):
    """
    Calculate correlations between SFS and Google Trends.
    
    CORRELATION EXPLAINED:
    Correlation measures the strength of relationship between two variables.
    - +1.0 = Perfect positive relationship (both move together)
    - 0.0 = No relationship
    - -1.0 = Perfect negative relationship (one goes up, other goes down)
    
    LAGGED CORRELATION:
    We also check if Google Trends LEADS the SFS data.
    A "lag" shifts one series back in time to see if past Google searches
    predict future fashion adoption.
    
    Example:
    - lag=1: Compare Google searches from month N with SFS from month N+1
    - If this correlation is higher, Google is a leading indicator
    
    Parameters:
    -----------
    sfs_df : DataFrame - Street Fashion Style data
    google_df : DataFrame - Google Trends data
    
    Returns:
    --------
    Dictionary containing correlation results
    """
    # Align datasets
    common_dates = sfs_df.index.intersection(google_df.index)
    sfs_aligned = sfs_df.loc[common_dates]
    google_aligned = google_df.loc[common_dates]
    
    print("=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Basic correlation (same time period)
    zara_corr = sfs_aligned['zara_frequency'].corr(google_aligned['zara_search_interest'])
    chanel_corr = sfs_aligned['chanel_frequency'].corr(google_aligned['chanel_search_interest'])
    
    print(f"\nSAME-TIME CORRELATION (SFS vs Google at same month):")
    print(f"  Zara:   {zara_corr:.3f}")
    print(f"  Chanel: {chanel_corr:.3f}")
    
    # Interpretation guide
    print("\n  Interpretation:")
    print("  0.0-0.3: Weak | 0.3-0.7: Moderate | 0.7-1.0: Strong")
    
    # Lagged correlation analysis
    print("\n" + "-" * 60)
    print("LAGGED CORRELATION (Does Google lead SFS?)")
    print("-" * 60)
    print("\nPositive lag = Google data from N months earlier")
    print("Higher correlation at lag > 0 suggests Google is a leading indicator\n")
    
    results = {
        'same_time': {'zara': zara_corr, 'chanel': chanel_corr},
        'lagged': {}
    }
    
    for lag in [1, 2, 3, 6]:
        # Shift Google data back (positive lag means Google earlier)
        # .shift(lag) moves data forward, so earlier Google values align with later SFS
        zara_lag_corr = sfs_aligned['zara_frequency'].corr(
            google_aligned['zara_search_interest'].shift(lag)
        )
        chanel_lag_corr = sfs_aligned['chanel_frequency'].corr(
            google_aligned['chanel_search_interest'].shift(lag)
        )
        
        results['lagged'][lag] = {'zara': zara_lag_corr, 'chanel': chanel_lag_corr}
        
        print(f"  Lag {lag} month(s): Zara={zara_lag_corr:.3f}, Chanel={chanel_lag_corr:.3f}")
    
    # Find optimal lag
    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    
    # Determine best lag for each trend
    for trend in ['zara', 'chanel']:
        all_corrs = [(0, results['same_time'][trend])]
        all_corrs += [(lag, results['lagged'][lag][trend]) for lag in results['lagged']]
        
        best_lag, best_corr = max(all_corrs, key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)
        
        if best_lag == 0:
            interpretation = "Google and SFS move together (contemporaneous)"
        else:
            interpretation = f"Google leads SFS by ~{best_lag} month(s)"
        
        print(f"\n  {trend.title()}:")
        print(f"    Best correlation: {best_corr:.3f} at lag {best_lag}")
        print(f"    Interpretation: {interpretation}")
    
    return results


def plot_weather_patterns(weather_df, save_path=None):
    """
    Visualize weather patterns over time and monthly seasonality.
    
    WHY WEATHER MATTERS FOR FASHION:
    - Temperature affects clothing choices (light vs warm clothes)
    - Rain might affect outdoor photography frequency
    - Seasonal patterns might correlate with fashion cycles
    
    Parameters:
    -----------
    weather_df : DataFrame - Monthly weather data
    save_path : str, optional - Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature over time
    axes[0, 0].plot(weather_df.index, weather_df['avg_temperature'], 
                    color='#E76F51', linewidth=1.5)
    axes[0, 0].set_title('Average Monthly Temperature', fontweight='bold')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Humidity over time
    axes[0, 1].plot(weather_df.index, weather_df['avg_humidity'], 
                    color='#2A9D8F', linewidth=1.5)
    axes[0, 1].set_title('Average Monthly Humidity', fontweight='bold')
    axes[0, 1].set_ylabel('Humidity (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rainfall over time
    axes[1, 0].bar(weather_df.index, weather_df['total_rainfall'], 
                   color='#457B9D', alpha=0.7, width=20)
    axes[1, 0].set_title('Total Monthly Rainfall', fontweight='bold')
    axes[1, 0].set_ylabel('Rainfall (mm)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cloud cover over time
    axes[1, 1].plot(weather_df.index, weather_df['avg_cloud_cover'], 
                    color='#6C757D', linewidth=1.5)
    axes[1, 1].set_title('Average Monthly Cloud Cover', fontweight='bold')
    axes[1, 1].set_ylabel('Cloud Cover (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_monthly_seasonality(weather_df, save_path=None):
    """
    Show average weather patterns by month (seasonality profile).
    
    SEASONALITY EXPLAINED:
    Seasonality is a repeating pattern that occurs at regular intervals.
    By averaging all January values, all February values, etc., we can
    see the typical seasonal pattern.
    
    This helps us understand:
    - When is it hot vs cold in California?
    - When does it rain most?
    - How might this affect fashion choices?
    
    Parameters:
    -----------
    weather_df : DataFrame - Monthly weather data
    save_path : str, optional - Path to save figure
    """
    # Add month column for grouping
    weather_copy = weather_df.copy()
    weather_copy['month'] = weather_copy.index.month
    
    # Calculate monthly averages across all years
    monthly_avg = weather_copy.groupby('month').mean()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(1, 13)
    width = 0.35
    
    # Normalize temperature to 0-100 scale for comparison with humidity
    temp_min = monthly_avg['avg_temperature'].min()
    temp_max = monthly_avg['avg_temperature'].max()
    temp_normalized = (monthly_avg['avg_temperature'] - temp_min) / (temp_max - temp_min) * 100
    
    bars1 = ax.bar(x - width/2, temp_normalized, width, 
                   label='Temperature (normalized 0-100)', color='#E76F51', alpha=0.8)
    bars2 = ax.bar(x + width/2, monthly_avg['avg_humidity'], width, 
                   label='Humidity (%)', color='#2A9D8F', alpha=0.8)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Value')
    ax.set_title('California Weather Seasonality Profile', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "=" * 50)
    print("MONTHLY WEATHER SUMMARY")
    print("=" * 50)
    print("\n" + monthly_avg.round(2).to_string())
    
    return monthly_avg


def print_summary_statistics(sfs_df, google_df, weather_df):
    """
    Print comprehensive summary statistics for all datasets.
    
    SUMMARY STATISTICS EXPLAINED:
    - count: Number of observations
    - mean: Average value
    - std: Standard deviation (spread/variability)
    - min/max: Range of values
    - 25%/50%/75%: Quartiles (distribution shape)
    
    These help us understand:
    - The scale of our data
    - Whether there are outliers
    - If data needs normalization
    """
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print("\n--- STREET FASHION STYLE ---")
    print(sfs_df.describe().round(2))
    
    print("\n--- GOOGLE TRENDS ---")
    print(google_df.describe().round(2))
    
    print("\n--- WEATHER ---")
    print(weather_df.describe().round(2))


def run_full_eda(sfs_df, google_df, weather_df, save_figures=True):
    """
    Run the complete EDA pipeline.
    
    Parameters:
    -----------
    sfs_df : DataFrame - Street Fashion Style data
    google_df : DataFrame - Google Trends data
    weather_df : DataFrame - Weather data
    save_figures : bool - Whether to save figures to files
    
    Returns:
    --------
    Dictionary containing all analysis results
    """
    results = {}
    
    # 1. Summary statistics
    print_summary_statistics(sfs_df, google_df, weather_df)
    
    # 2. Trend lifecycle visualization
    print("\n\n")
    peak_info = plot_trend_lifecycles(
        sfs_df, 
        save_path='plots/trend_lifecycles.png' if save_figures else None
    )
    results['peaks'] = peak_info
    
    # 3. SFS vs Google Trends comparison
    print("\n\n")
    plot_sfs_vs_google_trends(
        sfs_df, google_df,
        save_path='plots/sfs_vs_google.png' if save_figures else None
    )
    
    # 4. Correlation analysis
    print("\n\n")
    corr_results = analyze_correlations(sfs_df, google_df)
    results['correlations'] = corr_results
    
    # 5. Weather patterns
    print("\n\n")
    plot_weather_patterns(
        weather_df,
        save_path='plots/weather_patterns.png' if save_figures else None
    )
    
    # 6. Monthly seasonality
    print("\n\n")
    monthly_weather = plot_monthly_seasonality(
        weather_df,
        save_path='plots/weather_seasonality.png' if save_figures else None
    )
    results['monthly_weather'] = monthly_weather
    
    return results


# ============================================================
# MAIN - Run this file directly to perform EDA
# ============================================================
if __name__ == "__main__":
    import os
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    data = load_all_data(
        sfs_path='trend_counts_over_time.csv',
        google_path='google_trends.csv',
        weather_path='California_weather.csv'
    )
    
    # Run EDA
    results = run_full_eda(
        data['sfs'], 
        data['google'], 
        data['weather'],
        save_figures=True
    )
    
    print("\n\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print("\nKey findings saved in 'results' dictionary")
    print("Figures saved in 'plots/' directory")