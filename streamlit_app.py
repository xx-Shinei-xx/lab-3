import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, chisquare

def fit_and_test(data, data_mean):
    # Fit Poisson distribution
    poisson_params = poisson.fit(data, floc=0)
    poisson_pdf = poisson.pmf(data, *poisson_params[:-2])
    _, poisson_p_value = chisquare(data, poisson_pdf)

    # Fit Gaussian distribution
    gaussian_params = norm.fit(data)
    gaussian_pdf = norm.pdf(data, *gaussian_params)
    _, gaussian_p_value = chisquare(data, gaussian_pdf)
    
    return data_mean, poisson_p_value, gaussian_p_value

def main():
    st.title("Distribution Fitting App")
    st.write("Upload your CSV file")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())

        data_mean = df.mean().values[0]
        st.write(f"**Mean of Data**: {data_mean:.2f}")

        if st.button("Fit Distributions"):
            data_values = df.iloc[:, 0].values

            mean, poisson_p_value, gaussian_p_value = fit_and_test(data_values, data_mean)

            st.write("### Results")
            st.write(f"**Poisson Fit**: chi2 p-value = {poisson_p_value:.4f}")
            st.write(f"**Gaussian Fit**: chi2 p-value = {gaussian_p_value:.4f}")

if __name__ == "__main__":
    main()
