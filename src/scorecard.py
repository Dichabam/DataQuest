import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_names):
    """
    Extracts the weights (coefficients) from the Logistic Regression model.
    Because features are WoE-transformed (same scale), coefficients are
    directly comparable.
    """
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Importance': np.abs(coefficients),
    }).sort_values(by='Absolute_Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Coefficient'], color='skyblue')
    plt.xlabel('Coefficient Value (Impact on Default Risk)')
    plt.title('Feature Importance: What drives the model?')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.tight_layout()

    return importance_df.sort_values(by='Absolute_Importance', ascending=False)


def generate_scorecard(model, woe_engine, feature_names, target_score=600, target_odds=50, pdo=20):
    """
    Translates the model's coefficients into a traditional points-based Scorecard.

    Standard scaling formula (Anderson 2007):
        Score = Offset + Factor * ln(odds)
        Factor = PDO / ln(2)
        Offset = target_score - Factor * ln(target_odds)

    Points per bin:
        Points_i = -(beta_i * WoE_ij * Factor) - (alpha / n) * Factor + Offset / n

    FIX: Previously the intercept (alpha) was ignored entirely. The offset was
    split equally across features as `offset / n_features`, but this only
    accounts for the score anchor — it does NOT distribute the model intercept.
    The intercept shifts the log-odds baseline and must be factored in per
    feature. The corrected formula below distributes the intercept contribution
    equally across all features, which is the standard industry approach when
    a base score per feature is required.

    Reference: Siddiqi, N. (2006). Credit Risk Scorecards. Wiley.
    """
    factor = pdo / np.log(2)
    offset = target_score - (factor * np.log(target_odds))

    intercept = model.intercept_[0]
    coefficients = dict(zip(feature_names, model.coef_[0]))
    n_features = len(feature_names)

    # Intercept contribution split equally across features (standard approach)
    intercept_per_feature = intercept / n_features

    scorecard = []

    for feature in feature_names:
        original_feature = feature.replace('_woe', '')
        woe_dict = woe_engine.woe_dicts[original_feature]
        beta = coefficients[feature]

        for bin_name, woe_value in woe_dict.items():
            # Full formula:
            #   Points = -(beta * WoE * Factor)          <- variable part
            #            - (intercept/n * Factor)         <- intercept share (FIX)
            #            + (Offset / n)                   <- score anchor
            points = (
                -(beta * woe_value * factor)
                - (intercept_per_feature * factor)
                + (offset / n_features)
            )

            scorecard.append({
                'Feature': original_feature,
                'Bin': bin_name,
                'WoE': round(woe_value, 4),
                'Points': round(points),
            })

    return pd.DataFrame(scorecard)


def score_applicant(scorecard_df, woe_engine, applicant: dict) -> int:
    """
    Scores a single applicant dict using the generated scorecard.
    Returns an integer credit score.

    This is used by the Streamlit app to ensure the scorecard and the
    displayed credit score are derived from the SAME system.
    """
    total_points = 0

    for feature, woe_value in applicant.items():
        original_feature = feature.replace('_woe', '')
        feature_rows = scorecard_df[scorecard_df['Feature'] == original_feature]

        # Find the bin whose WoE is closest to this applicant's WoE value
        if feature_rows.empty:
            continue
        closest_idx = (feature_rows['WoE'] - woe_value).abs().idxmin()
        total_points += feature_rows.loc[closest_idx, 'Points']

    return int(total_points)