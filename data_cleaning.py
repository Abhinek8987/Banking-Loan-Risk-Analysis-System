import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data(input_path, output_path):
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Handle Missing Values
    # person_emp_length: Median is preferred as employment length can be skewed
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    # loan_int_rate: Median is safer against outliers
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    
    # 2. Convert Categorical to Numerical (One-Hot Encoding)
    # Categorical columns: person_home_ownership, loan_intent, loan_grade, cb_person_default_on_file
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 3. Remove Duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_count - len(df)} duplicate rows.")
    
    # 4. Detect and Treat Outliers (Business Logic)
    # Ages over 100 are likely errors in this context
    df = df[df['person_age'] <= 100]
    # Employment length cannot exceed age
    df = df[df['person_emp_length'] < df['person_age']]
    
    # 5. Feature Scaling
    # Scaling numerical features for Logistic Regression
    scaler = StandardScaler()
    numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_data('credit_risk_dataset.csv', 'cleaned_credit_risk.csv')
