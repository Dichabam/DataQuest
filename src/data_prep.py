import pandas as pd
import numpy as np

def load_and_split_data(filepath):
    """Loads the loan book and splits it based on the predefined 'set' column."""
    df = pd.read_csv(filepath)
    
    # Convert dates handling mixed formats (e.g., MM/DD/YYYY and YYYY-MM-DD)
    df['application_date'] = pd.to_datetime(df['application_date'], format='mixed')
    
    train_df = df[df['set'] == 'train'].copy()
    test_df = df[df['set'] == 'test'].copy()
    
    # Drop the set column as it's no longer needed
    train_df.drop(columns=['set'], inplace=True)
    test_df.drop(columns=['set'], inplace=True)
    
    return train_df, test_df

def clean_data(df, is_train=True, imputers=None):
    """
    Cleans the dataframe by handling missing values and outliers.
    Returns the cleaned dataframe and the imputers used (fitted on train).
    """
    cleaned_df = df.copy()
    
    if is_train:
        imputers = {
            'months_since_last_delinquency': cleaned_df['months_since_last_delinquency'].median(),
            'pct_accounts_current': cleaned_df['pct_accounts_current'].median(),
            # Cap extreme outliers for robust linear modelling (e.g., income > 99th percentile)
            'income_cap': cleaned_df['annual_income'].quantile(0.99)
        }
    
    # 1. Handle Missing Values
    # Missing delinquency implies no recent delinquency; fill with a high logical value (e.g., 999) 
    # to distinguish it, which our WoE binning will handle later.
    cleaned_df['months_since_last_delinquency'] = cleaned_df['months_since_last_delinquency'].fillna(999)
    cleaned_df['pct_accounts_current'] = cleaned_df['pct_accounts_current'].fillna(imputers['pct_accounts_current'])
    
    # 2. Outlier Treatment
    cleaned_df['annual_income'] = np.where(
        cleaned_df['annual_income'] > imputers['income_cap'], 
        imputers['income_cap'], 
        cleaned_df['annual_income']
    )
    
    # 3. String formatting
    cleaned_df['home_ownership'] = cleaned_df['home_ownership'].str.upper() # Standardize RENT/rent
    
    return cleaned_df, imputers

if __name__ == "__main__":
    # Test the pipeline
    train, test = load_and_split_data("../data/raw/loan_book.csv")
    train_clean, imputers = clean_data(train, is_train=True)
    test_clean, _ = clean_data(test, is_train=False, imputers=imputers)
    print(f"Train shape: {train_clean.shape}, Test shape: {test_clean.shape}")