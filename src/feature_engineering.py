import pandas as pd
import numpy as np

class WoE_Binning:
    def __init__(self, target_col='default_flag'):
        self.target_col = target_col
        self.woe_dicts = {}
        self.iv_scores = {}
        self.bins = {}

    def _calculate_woe_iv(self, df, feature, target):
        grouped = df.groupby(feature)[target].agg(['count', 'sum'])
        grouped.columns = ['Total', 'Bad']
        grouped['Good'] = grouped['Total'] - grouped['Bad']

        total_bad = grouped['Bad'].sum()
        total_good = grouped['Good'].sum()

        grouped['Dist_Good'] = np.maximum(grouped['Good'], 0.5) / total_good
        grouped['Dist_Bad'] = np.maximum(grouped['Bad'], 0.5) / total_bad

        grouped['WoE'] = np.log(grouped['Dist_Good'] / grouped['Dist_Bad'])
        grouped['IV'] = (grouped['Dist_Good'] - grouped['Dist_Bad']) * grouped['WoE']

        return grouped['WoE'].to_dict(), grouped['IV'].sum()

    def fit_transform_continuous(self, df, feature, q=5):
        df = df.copy()
        df[f'{feature}_binned'], bins = pd.qcut(
            df[feature], q=q, retbins=True, duplicates='drop'
        )
        df[f'{feature}_binned'] = df[f'{feature}_binned'].astype(str)
        self.bins[feature] = bins

        woe_dict, iv = self._calculate_woe_iv(df, f'{feature}_binned', self.target_col)
        self.woe_dicts[feature] = woe_dict
        self.iv_scores[feature] = iv

        df[f'{feature}_woe'] = df[f'{feature}_binned'].map(woe_dict)
        return df

    def fit_transform_categorical(self, df, feature):
        df = df.copy()
        woe_dict, iv = self._calculate_woe_iv(df, feature, self.target_col)
        self.woe_dicts[feature] = woe_dict
        self.iv_scores[feature] = iv
        df[f'{feature}_woe'] = df[feature].map(woe_dict)
        return df

    def transform(self, df, feature, is_continuous=False):
        df = df.copy()
        if is_continuous:
            df[f'{feature}_binned'] = pd.cut(
                df[feature], bins=self.bins[feature], include_lowest=True
            ).astype(str)
            df[f'{feature}_woe'] = df[f'{feature}_binned'].map(self.woe_dicts[feature]).fillna(0)
        else:
            df[f'{feature}_woe'] = df[feature].map(self.woe_dicts[feature]).fillna(0)
        return df

def engineer_features(train_df, test_df):
    woe_engine = WoE_Binning(target_col='default_flag')

    # FIX: Added the two missing variables that were cleaned in data_prep.py
    continuous_features = [
        'age', 'annual_income', 'credit_utilisation_pct', 'dti_ratio', 'loan_amount',
        'months_since_last_delinquency', 'pct_accounts_current'
    ]
    categorical_features = ['home_ownership', 'loan_purpose', 'region']

    for col in continuous_features:
        train_df = woe_engine.fit_transform_continuous(train_df, col, q=5)

    for col in categorical_features:
        train_df = woe_engine.fit_transform_categorical(train_df, col)

    for col in continuous_features:
        test_df = woe_engine.transform(test_df, col, is_continuous=True)

    for col in categorical_features:
        test_df = woe_engine.transform(test_df, col, is_continuous=False)

    return train_df, test_df, woe_engine