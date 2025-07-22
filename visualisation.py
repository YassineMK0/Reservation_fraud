import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv("reservations_cleaned.csv")

# Create output folder
output_folder = "paliz"
os.makedirs(output_folder, exist_ok=True)

# Add reservation hour for time-based plots
df['reservation_hour'] = pd.to_datetime(df['reservation_time'], errors='coerce').dt.hour

# --- NUMERICAL DISTRIBUTIONS ---
numerical_cols = ['nbr_place', 'number_of_voyageurs', 'account_age_days',
                  'payment_delay_days', 'total_payment_amount', 'payment_failures_count',
                  'satisfaction_client', 'tentatives_paiement', 'annulations_precedentes',
                  'modifications_reservation']

df[numerical_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()
plt.savefig(f"{output_folder}/numerical_distributions.png")
plt.close()

# --- CORRELATION MATRIX ---
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_cols + ['is_fraud']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig(f"{output_folder}/correlation_matrix.png")
plt.close()

# --- FRAUD VS STATUS ---
sns.countplot(data=df, x='Status', hue='is_fraud')
plt.title("Reservation Status vs Fraud")
plt.xticks(rotation=45)
plt.savefig(f"{output_folder}/status_vs_fraud.png")
plt.close()

# --- FRAUD VS PAYMENT STATUS ---
sns.countplot(data=df, x='payment_status', hue='is_fraud')
plt.title("Payment Status vs Fraud")
plt.savefig(f"{output_folder}/payment_status_vs_fraud.png")
plt.close()

# --- FRAUD VS REMINDER SENT ---
sns.countplot(data=df, x='reminderSent', hue='is_fraud')
plt.title("Reminder Sent vs Fraud")
plt.savefig(f"{output_folder}/reminder_vs_fraud.png")
plt.close()

# --- FRAUD VS NEWSLETTER ---
sns.countplot(data=df, x='newsletter_abonne', hue='is_fraud')
plt.title("Newsletter vs Fraud")
plt.savefig(f"{output_folder}/newsletter_vs_fraud.png")
plt.close()

# --- FRAUD VS EMAIL DOMAIN ---
top_domains = df['email_domain'].value_counts().nlargest(5).index
filtered_email = df[df['email_domain'].isin(top_domains)]
sns.countplot(data=filtered_email, x='email_domain', hue='is_fraud')
plt.title("Top Email Domains vs Fraud")
plt.savefig(f"{output_folder}/email_domain_vs_fraud.png")
plt.close()

# --- FRAUD VS COUNTRY ---
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Pays', hue='is_fraud')
plt.title("Country vs Fraud")
plt.savefig(f"{output_folder}/country_vs_fraud.png")
plt.close()

# --- FRAUD VS RESERVATION HOUR ---
sns.histplot(data=df, x='reservation_hour', hue='is_fraud', bins=24, kde=True, multiple='stack')
plt.title("Reservation Hour vs Fraud")
plt.xlabel("Hour")
plt.savefig(f"{output_folder}/reservation_hour_vs_fraud.png")
plt.close()

# --- ACCOUNT AGE VS FRAUD ---
sns.boxplot(data=df, x='is_fraud', y='account_age_days')
plt.title("Account Age vs Fraud")
plt.savefig(f"{output_folder}/account_age_vs_fraud.png")
plt.close()

print("✅ All EDA plots saved to folder: 'paliz'")
