import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and parse
df = pd.read_csv('data/SFS_metadata.csv')
df['time'] = df['time'].str.replace("Updated on ", "", regex=False)
df['date'] = pd.to_datetime(df['time'], errors='coerce')
df['tags'] = df['tags'].fillna("").str.lower()

# Tag matching
trend_items = {
    "zara_dress": ["zara", "dress"],
    "chanel_bag": ["chanel", "bag"]
}

for trend_label, keywords in trend_items.items():
    df[trend_label] = df['tags'].apply(
        lambda x: 1 if all(kw in x for kw in keywords) else 0)

# Create daily counts
daily = df.groupby('date')[list(trend_items.keys())].sum()
daily = daily.asfreq('D', fill_value=0)

# Aggregate to weekly
weekly = daily.resample('W').sum()

print("="*50)
print("WEEKLY DATA BEFORE SMOOTHING")
print("="*50)
print(f"Zara zeros: {(weekly['zara_dress'] == 0).sum()} / {len(weekly)}")
print(f"Chanel zeros: {(weekly['chanel_bag'] == 0).sum()} / {len(weekly)}")

# Apply smoothing (4-week rolling average for weekly data)
weekly['zara_dress_smooth'] = weekly['zara_dress'].rolling(4, min_periods=1).mean()
weekly['chanel_bag_smooth'] = weekly['chanel_bag'].rolling(4, min_periods=1).mean()

print("\n" + "="*50)
print("WEEKLY DATA AFTER 4-WEEK SMOOTHING")
print("="*50)
print(f"Zara zeros: {(weekly['zara_dress_smooth'] == 0).sum()} / {len(weekly)}")
print(f"Chanel zeros: {(weekly['chanel_bag_smooth'] == 0).sum()} / {len(weekly)}")

# Alternative: Interpolate zeros instead of rolling average
weekly['zara_dress_interp'] = weekly['zara_dress'].replace(0, np.nan).interpolate(method='linear').fillna(0)
weekly['chanel_bag_interp'] = weekly['chanel_bag'].replace(0, np.nan).interpolate(method='linear').fillna(0)

print("\n" + "="*50)
print("WEEKLY DATA AFTER INTERPOLATION")
print("="*50)
print(f"Zara zeros: {(weekly['zara_dress_interp'] == 0).sum()} / {len(weekly)}")
print(f"Chanel zeros: {(weekly['chanel_bag_interp'] == 0).sum()} / {len(weekly)}")

# Visualize comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Zara
axes[0].plot(weekly.index, weekly['zara_dress'], alpha=0.3, label='Raw', color='gray')
axes[0].plot(weekly.index, weekly['zara_dress_smooth'], label='4-week MA', color='blue')
axes[0].plot(weekly.index, weekly['zara_dress_interp'], label='Interpolated', color='green', linestyle='--')
axes[0].set_title('Zara Dress - Weekly Frequency')
axes[0].legend()
axes[0].set_ylabel('Count')

# Chanel
axes[1].plot(weekly.index, weekly['chanel_bag'], alpha=0.3, label='Raw', color='gray')
axes[1].plot(weekly.index, weekly['chanel_bag_smooth'], label='4-week MA', color='blue')
axes[1].plot(weekly.index, weekly['chanel_bag_interp'], label='Interpolated', color='green', linestyle='--')
axes[1].set_title('Chanel Bag - Weekly Frequency')
axes[1].legend()
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('plots/weekly_smoothing_comparison.png', dpi=150)
plt.show()

# Save the weekly data for modeling
weekly.to_csv('data/weekly_trend_counts.csv')
print("\nSaved: data/weekly_trend_counts.csv")
print(f"Shape: {weekly.shape}")
print(weekly.head(20))