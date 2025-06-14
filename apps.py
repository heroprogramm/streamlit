import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Basic Streamlit App with External Libraries")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Plot if numeric data exists
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_columns:
        col = st.selectbox("Choose a column to plot", numeric_columns)
        st.line_chart(df[col])
    else:
        st.warning("No numeric columns to plot.")
