import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_financial_data.csv")

plt.figure(figsize=(10,6))
sns.histplot(df['AccountBalance'], kde=True)
plt.title("Account Balance Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data=df, x='TransactionType')
plt.title("Transaction Type Distribution")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df, y='TransactionAmount')
plt.title("Transaction Amount Outliers")
plt.show()