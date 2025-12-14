import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('data/SFS_metadata.csv')

df['time'] = df['time'].str.replace("Updated on ", "", regex=False)
df['date'] = pd.to_datetime(df['time'], errors='coerce')

df['tags'] = df['tags'].fillna("").str.lower()

trend_items = {
    "zara_dress": ["zara", "dress"],
    "chanel_bag": ["chanel", "bag"]
}


for trend_label, keywords in trend_items.items():
    df[trend_label] = df['tags'].apply(
        lambda x: 1 if all(kw in x for kw in keywords) else 0)
    
daily = df.groupby('date')[list(trend_items.keys())].sum()


daily = daily.asfreq('D', fill_value=0)

smooth = daily['zara_dress'].rolling(7, min_periods=1).mean()
smooth = daily['chanel_bag'].rolling(7, min_periods=1).mean()
smoothed = sm.nonparametric.lowess(daily['zara_dress'], daily.index, frac=0.2)
smoothed = sm.nonparametric.lowess(daily['chanel_bag'], daily.index, frac=0.2)

daily.head(20)


