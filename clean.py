"""import pandas as pd

# Load the dataset
df = pd.read_csv("synthetic_reservations_5000_with_15percent_fraud.csv")

# Step 1: Convert 'Date_Reservation' to datetime
df['Date_Reservation'] = pd.to_datetime(df['Date_Reservation'])

# Step 2: Split Date_Reservation into separate components
df['reservation_date'] = df['Date_Reservation'].dt.date
df['reservation_time'] = df['Date_Reservation'].dt.time
df['reservation_year'] = df['Date_Reservation'].dt.year
df['reservation_month'] = df['Date_Reservation'].dt.month
df['reservation_day'] = df['Date_Reservation'].dt.day
df['reservation_hour'] = df['Date_Reservation'].dt.hour
df['reservation_minute'] = df['Date_Reservation'].dt.minute
df['reservation_dayofweek'] = df['Date_Reservation'].dt.dayofweek  # Monday=0

# Step 3: Convert boolean columns explicitly (if not already)
bool_columns = ['reminderSent', 'newsletter_abonne', 'suspicious_email_domain']
for col in bool_columns:
    df[col] = df[col].astype(bool)

# Step 4: Drop the original Date_Reservation column (optional)
# df.drop(columns=['Date_Reservation'], inplace=True)

# Step 5: Preview the updated dataframe
print(df.head())

# Step 6: Save to a new CSV file (optional)
df.to_csv("reservations_cleaned.csv", index=False)
print("✅ Cleaned and split data saved to 'synthetic_reservations_5000_with_15percent_fraud_cleaned.csv'")"""
import pandas as pd

# Load the dataset
df = pd.read_csv("reservations_cleaned.csv")

df.drop(columns=['reservation_year', 'reservation_day', 'reservation_minute'], inplace=True)


# Save the cleaned data
df.to_csv("reservations_cleaned.csv", index=False)
print("✅ Final cleaned dataset saved to 'reservations_cleaned.csv'")
