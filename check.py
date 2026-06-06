import pandas as pd

matches = pd.read_csv("data/matches.csv")
print("Matches shape:", matches.shape)
print(matches.head())
print(matches.columns.tolist())