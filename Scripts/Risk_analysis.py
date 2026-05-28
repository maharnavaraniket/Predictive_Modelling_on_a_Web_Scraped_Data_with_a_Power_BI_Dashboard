import pandas as pd

df = pd.read_csv("data/cleaned_financial_data.csv")

# Large withdrawals
large_withdrawals = df[
    (df['TransactionType'] == 'Withdrawal') &
    (df['TransactionAmount'] > 50000)
]

# Overdraft accounts
overdraft_accounts = df[df['AccountBalance'] < 0]

# Balance volatility
volatility = df.groupby('AccountID')['AccountBalance'].std()

print(large_withdrawals.head())
print(overdraft_accounts.head())
print(volatility.head())