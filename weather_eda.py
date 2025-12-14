import pandas as pd
df = pd.read_csv('data/California_weather.csv')
print(df.columns.tolist())
print(df.head())