import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/raw_financial_data.csv")

# Check missing values
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert transaction date
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

# Convert numerical columns
numeric_cols = ['TransactionAmount', 'AccountBalance',
                'RiskScore', 'CreditRating', 'TenureMonths']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Standardize categories
df['TransactionType'] = df['TransactionType'].str.title()
df['AccountType'] = df['AccountType'].str.title()

# Save cleaned data
df.to_csv("data/cleaned_financial_data.csv", index=False)

print("Data Cleaning Completed")