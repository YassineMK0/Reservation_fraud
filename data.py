import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
N = 5000  # total records
fraud_ratio = 0.15  # 15% fraud
fraud_count = int(N * fraud_ratio)
legit_count = N - fraud_count

def random_datetime(days_back=365):
    """Generate random datetime within specified days back from now"""
    start = datetime.now() - timedelta(days=days_back)
    random_seconds = random.randint(0, days_back * 24 * 3600)
    return start + timedelta(seconds=random_seconds)

def generate_rows(n, is_fraud):
    """Generate n rows of synthetic reservation data"""
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'tempmail.com', 'mailinator.com']
    countries = ['Tunisia', 'France', 'USA', 'Germany', 'Morocco']
    cities = ['Tunis', 'Paris', 'New York', 'Berlin', 'Casablanca']
    payment_modes = ['credit_card', 'paypal', 'bank_transfer', 'crypto']
    statuses = ['confirmé', 'annulé', 'en attente']
    
    data = []
    for i in range(n):
        # Date generation with fraud-specific patterns
        date_creation = random_datetime(days_back=30 if is_fraud else 365)
        date_reservation = date_creation + timedelta(days=random.randint(0, 30))
        
        # Reservation details
        nbr_place = random.randint(1, 3)
        number_of_voyageurs = random.randint(1, nbr_place)
        
        # Status with different distributions for fraud vs legit
        status = random.choices(
            statuses, 
            weights=[0.4, 0.4, 0.2] if is_fraud else [0.8, 0.1, 0.1]
        )[0]
        
        reminder_sent = random.choices([True, False], weights=[0.7, 0.3])[0]
        vol_id_frequency = random.randint(1, 100)
        
        # Email domain with fraud bias towards suspicious domains
        if is_fraud:
            email_domain = random.choices(
                ['tempmail.com', 'mailinator.com', 'gmail.com'], 
                weights=[0.45, 0.45, 0.1]
            )[0]
        else:
            email_domain = random.choices(
                domains, 
                weights=[0.6, 0.15, 0.15, 0.05, 0.05]
            )[0]
        
        email = f"user_{random.randint(1000, 9999)}@{email_domain}"
        country = random.choice(countries)
        city = random.choice(cities)
        newsletter = random.choice([True, False])
        satisfaction = random.choice([1, 2, 3, 4, 5])
        
        # Behavioral patterns (higher for fraud)
        annulations_precedentes = np.random.poisson(3 if is_fraud else 0.5)
        modifications_reservation = np.random.poisson(2 if is_fraud else 0.3)
        tentatives_paiement = np.random.poisson(3 if is_fraud else 0.2)
        
        # Payment details
        montant = nbr_place * random.uniform(50, 300)
        payment_status = 'échoué' if (is_fraud and random.random() < 0.4) else 'réussi'
        payment_failures_count = 1 if payment_status == 'échoué' else 0
        payment_mode = random.choice(payment_modes)
        date_paiement = date_reservation + timedelta(days=random.randint(0, 7))
        
        # Derived features
        reservation_hour = date_reservation.hour
        reservation_dayofweek = date_reservation.weekday()
        reservation_month = date_reservation.month
        account_age_days = (date_reservation - date_creation).days
        payment_delay_days = (date_paiement - date_reservation).days
        suspicious_email_domain = 1 if email_domain in ['tempmail.com', 'mailinator.com', '10minutemail.com'] else 0
        
        data.append({
            'reservation_id': f"res_{random.randint(100000, 999999)}",
            'user_id': f"user_{random.randint(1000, 9999)}",
            'Date_Reservation': date_reservation,
            'nbr_place': nbr_place,
            'number_of_voyageurs': number_of_voyageurs,
            'Status': status,
            'reminderSent': reminder_sent,
            'volId_frequency': vol_id_frequency,
            'reservation_hour': reservation_hour,
            'reservation_dayofweek': reservation_dayofweek,
            'reservation_month': reservation_month,
            'account_age_days': account_age_days,
            'payment_delay_days': payment_delay_days,
            'total_payment_amount': round(montant, 2),
            'payment_failures_count': payment_failures_count,
            'payment_status': payment_status,
            'Pays': country,
            'Ville': city,
            'newsletter_abonne': newsletter,
            'satisfaction_client': satisfaction,
            'annulations_precedentes': annulations_precedentes,
            'modifications_reservation': modifications_reservation,
            'tentatives_paiement': tentatives_paiement,
            'email_domain': email_domain,
            'suspicious_email_domain': suspicious_email_domain,
            'is_fraud': is_fraud
        })
    return data

def generate_dataset():
    """Generate complete dataset with fraud and legitimate records"""
    print(f"Generating {N} records with {fraud_ratio*100}% fraud rate...")
    print(f"Fraud records: {fraud_count}")
    print(f"Legitimate records: {legit_count}")
    
    # Generate the datasets
    fraud_data = generate_rows(fraud_count, is_fraud=1)
    legit_data = generate_rows(legit_count, is_fraud=0)
    
    # Combine and shuffle the data
    all_data = pd.DataFrame(fraud_data + legit_data).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return all_data

def save_dataset(df, filename="synthetic_reservations_5000_with_15percent_fraud.csv"):
    """Save dataset to CSV file"""
    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved to {filename}")
    return filename

def analyze_dataset(df):
    """Provide basic analysis of the generated dataset"""
    print("\n📊 Dataset Analysis:")
    print(f"Total records: {len(df)}")
    print(f"Fraud records: {df['is_fraud'].sum()}")
    print(f"Legitimate records: {len(df) - df['is_fraud'].sum()}")
    print(f"Actual fraud rate: {df['is_fraud'].mean():.2%}")
    
    print("\n🏷️ Status distribution:")
    status_by_fraud = df.groupby(['is_fraud', 'Status']).size().unstack(fill_value=0)
    print(status_by_fraud)
    
    print("\n💳 Payment status by fraud:")
    payment_by_fraud = df.groupby(['is_fraud', 'payment_status']).size().unstack(fill_value=0)
    print(payment_by_fraud)
    
    print("\n📧 Suspicious email domains:")
    suspicious_emails = df[df['suspicious_email_domain'] == 1]
    print(f"Records with suspicious email domains: {len(suspicious_emails)}")
    print(f"Fraud rate in suspicious emails: {suspicious_emails['is_fraud'].mean():.2%}")
    
    return df.describe()

def main():
    """Main execution function"""
    print("🚀 Starting synthetic reservation data generation...")
    
    # Generate dataset
    dataset = generate_dataset()
    
    # Save to file
    filename = save_dataset(dataset)
    
    # Analyze dataset
    stats = analyze_dataset(dataset)
    
    # Display first few rows
    print("\n📋 First 5 records:")
    print(dataset.head())
    
    print("\n✨ Generation complete!")
    print(f"Dataset ready for machine learning at: {filename}")
    
    return dataset

# Execute if run as main script
if __name__ == "__main__":
    df = main()