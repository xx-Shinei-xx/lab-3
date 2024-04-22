import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, chisquare

def main():
    st.title("Poisson and Gaussian Distribution Adjustments")

    # Load data from CSV files
    data1_file_path = 'data1.csv'
    data2_file_path = 'data2.csv'

    data1 = pd.read_csv(data1_file_path, header=None, squeeze=True)
    data2 = pd.read_csv(data2_file_path, header=None, squeeze=True)

    # Poisson distribution adjustment
    lambda_air = data1.mean()
    lambda_cesium = data2.mean()

    poisson_air_chisq = chisquare(data1, poisson.pmf(data1, lambda_air))
    poisson_cesium_chisq = chisquare(data2, poisson.pmf(data2, lambda_cesium))

    # Gaussian distribution adjustment
    mu_air = data1.mean()
    sigma_air = data1.std()
    mu_cesium = data2.mean()
    sigma_cesium = data2.std()

    gaussian_air_chisq = chisquare(data1, norm.pdf(data1, mu_air, sigma_air))
    gaussian_cesium_chisq = chisquare(data2, norm.pdf(data2, mu_cesium, sigma_cesium))

    # Display results
    st.header("Poisson Distribution Adjustment")
    st.write(f"Air data: chi-square statistic = {poisson_air_chisq.statistic:.2f}, p-value = {poisson_air_chisq.pvalue:.2e}")
    st.write(f"Cesium data: chi-square statistic = {poisson_cesium_chisq.statistic:.2f}, p-value = {poisson_cesium_chisq.pvalue:.2e}")

    st.header("Gaussian Distribution Adjustment")
    st.write(f"Air data: chi-square statistic = {gaussian_air_chisq.statistic:.2f}, p-value = {gaussian_air_chisq.pvalue:.2e}")
    st.write(f"Cesium data: chi-square statistic = {gaussian_cesium_chisq.statistic:.2f}, p-value = {gaussian_cesium_chisq.pvalue:.2e}")

    st.header("Best Adjustment")
    if poisson_air_chisq.pvalue > gaussian_air_chisq.pvalue:
        st.write("For the air data, the Poisson distribution adjustment provides a better fit.")
    else:
        st.write("For the air data, the Gaussian distribution adjustment provides a better fit.")

    if poisson_cesium_chisq.pvalue > gaussian_cesium_chisq.pvalue:
        st.write("For the cesium data, the Poisson distribution adjustment provides a better fit.")
    else:
        st.write("For the cesium data, the Gaussian distribution adjustment provides a better fit.")

if __name__ == "__main__":
    main()
