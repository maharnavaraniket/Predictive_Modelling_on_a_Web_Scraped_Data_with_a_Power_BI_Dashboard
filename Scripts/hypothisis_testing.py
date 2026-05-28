import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv("data/cleaned_financial_data.csv")

high_volume = df[df['TransactionAmount'] > 50000]['AccountBalance']
low_volume = df[df['TransactionAmount'] <= 50000]['AccountBalance']

t_stat, p_value = ttest_ind(high_volume, low_volume)

print("T-statistic:", t_stat)
print("P-value:", p_value)