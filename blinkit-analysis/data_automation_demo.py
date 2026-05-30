print(">>> SCRIPT STARTED <<<")

import pandas as pd

# Load order data
df = pd.read_csv("blinkit_orders.csv")

# Basic cleaning
df['order_date'] = pd.to_datetime(df['order_date'])
df_clean = df.dropna()

# Simple analysis
total_orders = len(df_clean)
total_revenue = df_clean['order_total'].sum()

print("Total Orders:", total_orders)
print("Total Revenue:", total_revenue)

# Save output
df_clean.to_csv("cleaned_orders.csv", index=False)

print("Cleaned file saved as cleaned_orders.csv")