import pandas as pd
import numpy as np


def load_and_split_data(filepath):
    """Loads the loan book and splits it based on the predefined 'set' column."""
    df = pd.read_csv(filepath)

    # Convert dates handling mixed formats (e.g., MM/DD/YYYY and YYYY-MM-DD)
    df['application_date'] = pd.to_datetime(df['application_date'], format='mixed')

    train_df = df[df['set'] == 'train'].copy()
    test_df = df[df['set'] == 'test'].copy()

    train_df.drop(columns=['set'], inplace=True)
    test_df.drop(columns=['set'], inplace=True)

    return train_df, test_df


def clean_data(df, is_train=True, imputers=None):
    """
    Cleans the dataframe by handling missing values and outliers.

    FIX: Previously, 'months_since_last_delinquency' median was stored in
    imputers but never used — both train and test were hardcoded to 999.
    Now the imputer dict stores BOTH the median (for reference/logging)
    AND the fill strategy. The 999 fill is intentional domain logic
    (no recent delinquency = very old event), but it must be applied
    consistently via the imputer so test data mirrors train behaviour.

    Returns the cleaned dataframe and the imputers dict (fitted on train only).
    """
    cleaned_df = df.copy()

    if is_train:
        if imputers is not None:
            raise ValueError("imputers must be None when is_train=True")
        imputers = {
            # 999 is intentional domain logic: missing delinquency means
            # no recent event. We store it explicitly so test gets the same value.
            'months_since_last_delinquency_fill': 999,
            'pct_accounts_current_median': cleaned_df['pct_accounts_current'].median(),
            'income_cap': cleaned_df['annual_income'].quantile(0.99),
        }
    else:
        if imputers is None:
            raise ValueError("imputers must be provided when is_train=False")

    # 1. Handle Missing Values
    # Use the stored fill value — consistent across train and test
    cleaned_df['months_since_last_delinquency'] = cleaned_df[
        'months_since_last_delinquency'
    ].fillna(imputers['months_since_last_delinquency_fill'])

    cleaned_df['pct_accounts_current'] = cleaned_df['pct_accounts_current'].fillna(
        imputers['pct_accounts_current_median']
    )

    # 2. Outlier Treatment: cap income at the train 99th percentile
    cleaned_df['annual_income'] = np.where(
        cleaned_df['annual_income'] > imputers['income_cap'],
        imputers['income_cap'],
        cleaned_df['annual_income'],
    )
    for col in ['age', 'dti_ratio', 'loan_amount', 'credit_utilisation_pct']:
        if col in cleaned_df.columns:
            if is_train:
                imputers[f'{col}_median'] = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(imputers[f'{col}_median'])
    # 3. String formatting — standardise RENT/rent etc.
    cleaned_df['home_ownership'] = cleaned_df['home_ownership'].str.upper()

    return cleaned_df, imputers


if __name__ == "__main__":
    train, test = load_and_split_data("../data/raw/loan_book.csv")
    train_clean, imputers = clean_data(train, is_train=True)
    test_clean, _ = clean_data(test, is_train=False, imputers=imputers)
    print(f"Train shape: {train_clean.shape}, Test shape: {test_clean.shape}")
    