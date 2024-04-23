import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, norm
from scipy.stats import chi2

# Streamlit app title
st.title("Poisson and Gaussian Distribution Fitting")

# File upload section
data1_file = st.file_uploader("Upload data1.csv", type="csv")
data2_file = st.file_uploader("Upload data2.csv", type="csv")

if data1_file is not None and data2_file is not None:
    # Read data from uploaded CSV files
    data1 = pd.read_csv(data1_file, header=None, names=['values'])
    data2 = pd.read_csv(data2_file, header=None, names=['values'])

    # Convert strings to numbers and handle non-convertible values
    data1['values'] = pd.to_numeric(data1['values'], errors='coerce')
    data2['values'] = pd.to_numeric(data2['values'], errors='coerce')

    # Drop rows with non-numeric values
    data1 = data1.dropna(subset=['values'])
    data2 = data2.dropna(subset=['values'])

    # Poisson Fit for data1
    data1_mean = data1['values'].mean()
    poisson_params = poisson.fit(data1['values'].values, floc=0)
    poisson_pdf = poisson.pmf(data1['values'], poisson_params[:-2])
    poisson_chi2, poisson_p_value1 = chi2.cdft(poisson_pdf, data1['values'])

    # Gaussian Fit for data1
    gaussian_params = norm.fit(data1['values'].values)
    gaussian_pdf = norm.pdf(data1['values'], gaussian_params[0], gaussian_params[1])
    gaussian_chi2, gaussian_p_value1 = chi2.cdft(gaussian_pdf, data1['values'])

    # Poisson Fit for data2
    data2_mean = data2['values'].mean()
    poisson_params = poisson.fit(data2['values'].values, floc=0)
    poisson_pdf = poisson.pmf(data2['values'], poisson_params[:-2])
    poisson_chi2, poisson_p_value2 = chi2.cdft(poisson_pdf, data2['values'])

    # Gaussian Fit for data2
    gaussian_params = norm.fit(data2['values'].values)
    gaussian_pdf = norm.pdf(data2['values'], gaussian_params[0], gaussian_params[1])
    gaussian_chi2, gaussian_p_value2 = chi2.cdft(gaussian_pdf, data2['values'])

    # Display results
    st.subheader("Results for data1.csv")
    st.write(f"Poisson fit: mean = {data1_mean:.2f}, chi2 = {poisson_chi2:.2f}, p-value = {poisson_p_value1:.4f}")
    st.write(f"Gaussian fit: mean = {gaussian_params[0]:.2f}, std = {gaussian_params[1]:.2f}, chi2 = {gaussian_chi2:.2f}, p-value = {gaussian_p_value1:.4f}")

    st.subheader("Results for data2.csv")
    st.write(f"Poisson fit: mean = {data2_mean:.2f}, chi2 = {poisson_chi2:.2f}, p-value = {poisson_p_value2:.4f}")
    st.write(f"Gaussian fit: mean = {gaussian_params[0]:.2f}, std = {gaussian_params[1]:.2f}, chi2 = {gaussian_chi2:.2f}, p-value = {gaussian_p_value2:.4f}")

else:
    st.warning("Please upload both data1.csv and data2.csv files.")
