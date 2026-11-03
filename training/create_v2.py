
import pandas as pd

input_path = "data/raw/breast_cancer.csv"

df = pd.read_csv(input_path)

print("Original rows:", len(df))

# Remove 30 rows
df_v2 = df.iloc[:-30].copy()

df_v2.to_csv(input_path, index=False)

print("Second version created!")
print("New rows:", len(df_v2))
print("Features:", len(df_v2.columns) - 1)
print("Total columns:", len(df_v2.columns))
