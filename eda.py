import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda(input_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(input_path)
    # Using the raw data for EDA to have readable labels, 
    # but since we already have cleaned_credit_risk.csv, let's use the original for unscaled values if needed.
    # Actually, the user wants EDA on the dataset. Let's use the original but handle missing for plotting.
    raw_df = pd.read_csv('credit_risk_dataset.csv')
    
    # 1. Income vs Default
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='loan_status', y='person_income', data=raw_df)
    plt.title('Income vs Loan Default (0=No, 1=Yes)')
    plt.yscale('log') # Log scale due to high income variation
    plt.savefig(os.path.join(output_dir, 'income_vs_default.png'))
    plt.close()

    # 2. Credit History vs Default
    # Note: cb_person_default_on_file is categorical
    plt.figure(figsize=(8, 5))
    sns.countplot(x='cb_person_default_on_file', hue='loan_status', data=raw_df)
    plt.title('Previous Default History vs Current Loan Status')
    plt.savefig(os.path.join(output_dir, 'credit_hist_vs_default.png'))
    plt.close()

    # 3. Loan Amount Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(raw_df['loan_amnt'], bins=30, kde=True)
    plt.title('Distribution of Loan Amounts')
    plt.savefig(os.path.join(output_dir, 'loan_amount_dist.png'))
    plt.close()

    # 4. Correlation Heatmap (using numerical columns from cleaned data)
    cleaned_df = pd.read_csv('cleaned_credit_risk.csv')
    plt.figure(figsize=(12, 10))
    sns.heatmap(cleaned_df.corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    run_eda('cleaned_credit_risk.csv', 'eda_plots')
