import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, chisquare
import requests

def load_data(file_url):
    response = requests.get(file_url)
    if response.status_code == 200:
        data = response.content.decode('utf-8').split('\n')
        return [int(value) for value in data if value]
    else:
        return None

def main():
    st.title("Poisson and Gaussian Distribution Adjustments")

    # GitHub repository URL
    repo_url = "https://raw.githubusercontent.com/your_username/your_repo/main/"

    # Load data from GitHub
    data1_url = repo_url + "data1.csv"
    data2_url = repo_url + "data2.csv"

    data1 = load_data(data1_url)
    data2 = load_data(data2_url)

    if data1 and data2:
        data1 = pd.Series(data1)
        data2 = pd.Series(data2)

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
        st.write(f"Air data: chi-square statistic = {poisson_air_chisq[0]:.2f}, p-value = {poisson_air_chisq[1]:.2e}")
        st.write(f"Cesium data: chi-square statistic = {poisson_cesium_chisq[0]:.2f}, p-value = {poisson_cesium_chisq[1]:.2e}")

        st.header("Gaussian Distribution Adjustment")
        st.write(f"Air data: chi-square statistic = {gaussian_air_chisq[0]:.2f}, p-value = {gaussian_air_chisq[1]:.2e}")
        st.write(f"Cesium data: chi-square statistic = {gaussian_cesium_chisq[0]:.2f}, p-value = {gaussian_cesium_chisq[1]:.2e}")

        st.header("Best Adjustment")
        if poisson_air_chisq[1] > gaussian_air_chisq[1]:
            st.write("For the air data, the Poisson distribution adjustment provides a better fit.")
        else:
            st.write("For the air data, the Gaussian distribution adjustment provides a better fit.")

        if poisson_cesium_chisq[1] > gaussian_cesium_chisq[1]:
            st.write("For the cesium data, the Poisson distribution adjustment provides a better fit.")
        else:
            st.write("For the cesium data, the Gaussian distribution adjustment provides a better fit.")

    else:
        st.warning("Error loading data from GitHub.")

if __name__ == "__main__":
    main()
    
