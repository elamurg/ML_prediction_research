"""
Transform Daily Google Trends Data to Weekly

This script:
1. Loads daily Google Trends data
2. Aggregates to weekly frequency
3. Applies a 4-week moving average for smoothing
4. Saves the processed data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load daily data
df = pd.read_csv('data/google_trends_weekly.csv', parse_dates=['date'])
df = df.set_index('date')

print("="*60)
print("ORIGINAL DAILY DATA")
print("="*60)
print(f"Rows: {len(df)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Step 1: Aggregate daily to weekly (using mean)
weekly = df.resample('W').mean()

print("\n" + "="*60)
print("AFTER WEEKLY AGGREGATION")
print("="*60)
print(f"Rows: {len(weekly)}")
print(f"Zara zeros: {(weekly['zara_search_interest'] == 0).sum()}")
print(f"Chanel zeros: {(weekly['chanel_search_interest'] == 0).sum()}")

# Step 2: Apply 4-week moving average
weekly['zara_search_interest_smooth'] = weekly['zara_search_interest'].rolling(4, min_periods=1).mean()
weekly['chanel_search_interest_smooth'] = weekly['chanel_search_interest'].rolling(4, min_periods=1).mean()

print("\n" + "="*60)
print("AFTER 4-WEEK MOVING AVERAGE")
print("="*60)
print(f"Zara zeros (smoothed): {(weekly['zara_search_interest_smooth'] == 0).sum()}")
print(f"Chanel zeros (smoothed): {(weekly['chanel_search_interest_smooth'] == 0).sum()}")

# Keep only smoothed columns, rename for clarity
weekly_final = weekly[['zara_search_interest_smooth', 'chanel_search_interest_smooth']].copy()
weekly_final.columns = ['zara_search_interest', 'chanel_search_interest']

# Print sample
print("\n" + "="*60)
print("SAMPLE DATA (first 20 weeks)")
print("="*60)
print(weekly_final.head(20).round(2))

print("\n" + "="*60)
print("SAMPLE DATA (2012 - peak period)")
print("="*60)
print(weekly_final['2012-01':'2012-06'].round(2))

# Summary stats
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print("\nZara Search Interest:")
print(weekly_final['zara_search_interest'].describe().round(2))
print("\nChanel Search Interest:")
print(weekly_final['chanel_search_interest'].describe().round(2))

# Save
output_path = 'data/google_trends_weekly_smoothed.csv'
weekly_final.to_csv(output_path)
print(f"\nSaved to: {output_path}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Zara - Raw weekly vs Smoothed
axes[0, 0].plot(weekly.index, weekly['zara_search_interest'], 
                color='gray', alpha=0.5, linewidth=1, label='Raw Weekly')
axes[0, 0].plot(weekly.index, weekly['zara_search_interest_smooth'], 
                color='#2A9D8F', linewidth=2, label='4-week MA')
axes[0, 0].set_title('Zara Dress - Weekly Google Trends', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Search Interest')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Chanel - Raw weekly vs Smoothed
axes[0, 1].plot(weekly.index, weekly['chanel_search_interest'], 
                color='gray', alpha=0.5, linewidth=1, label='Raw Weekly')
axes[0, 1].plot(weekly.index, weekly['chanel_search_interest_smooth'], 
                color='#E76F51', linewidth=2, label='4-week MA')
axes[0, 1].set_title('Chanel Bag - Weekly Google Trends', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Search Interest')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Zara - Final smoothed
axes[1, 0].plot(weekly_final.index, weekly_final['zara_search_interest'], 
                color='#2A9D8F', linewidth=2)
axes[1, 0].set_title('Zara Dress - Final (4-week MA)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Search Interest')
axes[1, 0].grid(True, alpha=0.3)

# Chanel - Final smoothed
axes[1, 1].plot(weekly_final.index, weekly_final['chanel_search_interest'], 
                color='#E76F51', linewidth=2)
axes[1, 1].set_title('Chanel Bag - Final (4-week MA)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Search Interest')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/google_trends_weekly_transformation.png', dpi=150)
print(f"Plot saved to: plots/google_trends_weekly_transformation.png")
plt.show()