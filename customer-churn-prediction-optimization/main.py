import pandas as pd

df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)