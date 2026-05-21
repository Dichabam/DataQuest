# DataQuest 2026: Interpretable Credit Modelling

## Executive Summary
This repository contains a production-grade, interpretable credit risk modelling pipeline and interactive Business Decision Dashboard. Built to satisfy strict regulatory constraints, the decision engine relies exclusively on Logistic Regression, enhanced through careful Weight of Evidence (WoE) feature engineering and Points-to-Double-Odds (PDO) scorecard generation.

Moving beyond standard ML metrics (AUC/ROC), this project bridges the gap between data science and business strategy by implementing dynamic P&L calculations, Macro-Economic Stress Testing, Population Stability Index (PSI) tracking, and an automated Regulatory Fairness Audit.

## Project Structure

```text
DataQuest/
│
├── app/
│   └── main_app.py              # The Streamlit Interactive Dashboard
│
├── data/
│   └── raw/
│       └── loan_book.csv        # Simulated historical loan dataset (Not included in repo)
│
├── src/
│   ├── data_prep.py             # Data loading, cleaning, and missing value imputation
│   ├── feature_engineering.py   # Weight of Evidence (WoE) & Information Value (IV) engine
│   ├── model_training.py        # Logistic Regression with 5-Fold Cross Validation (CV)
│   ├── scorecard.py             # Translates coefficients into a standard PDO Scorecard
│   └── business_logic.py        # P&L, Fairness Audits, PSI, and RF Shadow Modelling
│
├── test_pipeline.py             # End-to-end integration tests and validation assertions
├── requirements.txt             # Python dependencies
├── project_summary.txt          # Presentation script and slide outlines
└── README.md                    # Project documentation