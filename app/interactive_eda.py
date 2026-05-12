import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Credit Risk Exploratory Analysis")
st.sidebar.header("Filter Application Data")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/loan_book.csv")

df = load_data()

# Interactive Filters
selected_purpose = st.sidebar.selectbox("Loan Purpose", df['loan_purpose'].unique())
filtered_df = df[df['loan_purpose'] == selected_purpose]

# Visualisation
st.subheader(f"Default Rate by Home Ownership (Purpose: {selected_purpose})")
fig, ax = plt.subplots()
sns.barplot(data=filtered_df, x='home_ownership', y='default_flag', errorbar=None, ax=ax)
ax.set_ylabel("Probability of Default")
st.pyplot(fig)