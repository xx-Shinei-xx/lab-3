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

    # Load data from CSV files
    data1 = pd.read_csv('data1.csv', header=None, names=['values'])
    data2 = pd.read_csv('data2.csv', header=None, names=['values'])

    st.write("### Data 1 Preview")
    st.write(data1.head())
    st.write(f"**Mean of Data 1**: {data1['values'].mean():.2f}")

    st.write("### Data 2 Preview")
    st.write(data2.head())
    st.write(f"**Mean of Data 2**: {data2['values'].mean():.2f}")

    if st.button("Fit Distributions"):
        st.write("### Results for Data 1")
        data1_values = data1['values'].values
        mean1, poisson_p_value1, gaussian_p_value1 = fit_and_test(data1_values, data1['values'].mean())
        st.write(f"**Poisson Fit**: chi2 p-value = {poisson_p_value1:.4f}")
        st.write(f"**Gaussian Fit**: chi2 p-value = {gaussian_p_value1:.4f}")

        st.write("### Results for Data 2")
        data2_values = data2['values'].values
        mean2, poisson_p_value2, gaussian_p_value2 = fit_and_test(data2_values, data2['values'].mean())
        st.write(f"**Poisson Fit**: chi2 p-value = {poisson_p_value2:.4f}")
        st.write(f"**Gaussian Fit**: chi2 p-value = {gaussian_p_value2:.4f}")

if __name__ == "__main__":
    main()
