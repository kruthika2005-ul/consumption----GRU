import pandas as pd

# Read original dataset
df = pd.read_csv(
    "data/Electricity_consumption.csv",
    sep=",",          # Use ";" instead if your original file uses semicolons
    low_memory=False
)

# Keep first 100000 rows
df_small = df.head(100000)

# Save smaller dataset
df_small.to_csv(
    "data/Electricity_consumption_small.csv",
    index=False
)

print("Small dataset created successfully!")
print(df_small.shape)