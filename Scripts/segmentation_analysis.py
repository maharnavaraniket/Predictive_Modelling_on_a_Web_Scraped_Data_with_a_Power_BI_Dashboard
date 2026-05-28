import pandas as pd

df = pd.read_csv("data/cleaned_financial_data.csv")

customer_activity = df.groupby('CustomerID').size().reset_index(name='TransactionCount')

def segment_customer(x):
    if x >= 10:
        return 'High'
    elif x >= 5:
        return 'Medium'
    return 'Low'

customer_activity['Segment'] = customer_activity['TransactionCount'].apply(segment_customer)

print(customer_activity.head())

customer_activity.to_csv("outputs/customer_segments.csv", index=False)