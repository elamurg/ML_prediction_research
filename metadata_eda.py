import pandas as pd
import numpy as np

# Load and parse
df = pd.read_csv('data/SFS_metadata.csv')
df['time'] = df['time'].str.replace("Updated on ", "", regex=False)
df['date'] = pd.to_datetime(df['time'], errors='coerce')
df['tags'] = df['tags'].fillna("").str.lower()

print("="*50)
print("FULL DATASET OVERVIEW")
print("="*50)
print(f"Total records: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Tag matching
trend_items = {
    "zara_dress": ["zara", "dress"],
    "chanel_bag": ["chanel", "bag"]
}

for trend_label, keywords in trend_items.items():
    df[trend_label] = df['tags'].apply(
        lambda x: 1 if all(kw in x for kw in keywords) else 0)

# Filter to only matching records
zara_df = df[df['zara_dress'] == 1].copy()
chanel_df = df[df['chanel_bag'] == 1].copy()

print("\n" + "="*50)
print("ZARA DRESS")
print("="*50)
print(f"Total matches: {len(zara_df)}")
print(f"Date range: {zara_df['date'].min()} to {zara_df['date'].max()}")
print(f"Unique dates: {zara_df['date'].nunique()}")

# Daily and weekly counts for Zara
zara_daily = zara_df.groupby('date').size().asfreq('D', fill_value=0)
zara_weekly = zara_daily.resample('W').sum()
zara_monthly = zara_daily.resample('M').sum()

print(f"\nDaily samples: {len(zara_daily)}")
print(f"Days with posts > 0: {(zara_daily > 0).sum()}")
print(f"Weekly samples: {len(zara_weekly)}")
print(f"Weeks with posts > 0: {(zara_weekly > 0).sum()}")
print(f"Monthly samples: {len(zara_monthly)}")

print("\n" + "="*50)
print("CHANEL BAG")
print("="*50)
print(f"Total matches: {len(chanel_df)}")
print(f"Date range: {chanel_df['date'].min()} to {chanel_df['date'].max()}")
print(f"Unique dates: {chanel_df['date'].nunique()}")

# Daily and weekly counts for Chanel
chanel_daily = chanel_df.groupby('date').size().asfreq('D', fill_value=0)
chanel_weekly = chanel_daily.resample('W').sum()
chanel_monthly = chanel_daily.resample('M').sum()

print(f"\nDaily samples: {len(chanel_daily)}")
print(f"Days with posts > 0: {(chanel_daily > 0).sum()}")
print(f"Weekly samples: {len(chanel_weekly)}")
print(f"Weeks with posts > 0: {(chanel_weekly > 0).sum()}")
print(f"Monthly samples: {len(chanel_monthly)}")

print("\n" + "="*50)
print("SAMPLE DATA")
print("="*50)
print("\nZara weekly (first 20):")
print(zara_weekly.head(20))

print("\nChanel weekly (first 20):")
print(chanel_weekly.head(20))