import pandas as pd
data = pd.read_csv('data.csv', parse_dates=["day"])
data.set_index('day', inplace=True)
print(data)
print("******************************************************")

#new_df = data.fillna(method="ffill")
new_df = data.dropna()
#new_df = data.interpolate(method="time")
print(new_df)


print("******************************************************")

df = pd.read_csv('data2.csv')
print(df)
print("******************************************************")

max_height = df["height"].quantile(0.95)
min_height = df["height"].quantile(0.05)
print(df[(df['height'] < max_height) & (df['height'] > min_height)])
