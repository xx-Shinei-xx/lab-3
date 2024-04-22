import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, chisquare

def main():
    st.title("Poisson and Gaussian Distribution Adjustments")

    # Upload data files
    data1 = st.file_uploader("Upload data1.csv", type="csv")
    data2 = st.file_uploader("Upload data2.csv", type="csv")

    if data1 and data2:
        # Read data as strings
        data1 = pd.read_csv(data1, names=['decays_air'], header=None, dtype=str)
        data2 = pd.read_csv(data2, names=['decays_cesium'], header=None, dtype=str)

        # Convert to numeric
        data1['decays_air'] = data1['decays_air'].apply(lambda x: int(x))
        data2['decays_cesium'] = data2['decays_cesium'].apply(lambda x: int(x))

        # Poisson distribution adjustment
        lambda_air = data1['decays_air'].mean()
        lambda_cesium = data2['decays_cesium'].mean()

        poisson_air_chisq = chisquare(data1['decays_air'], poisson.pmf(data1['decays_air'], lambda_air))
        poisson_cesium_chisq = chisquare(data2['decays_cesium'], poisson.pmf(data2['decays_cesium'], lambda_cesium))

        # Gaussian distribution adjustment
        mu_air = data1['decays_air'].mean()
        sigma_air = data1['decays_air'].std()
        mu_cesium = data2['decays_cesium'].mean()
        sigma_cesium = data2['decays_cesium'].std()

        gaussian_air_chisq = chisquare(data1['decays_air'], norm.pdf(data1['decays_air'], mu_air, sigma_air))
        gaussian_cesium_chisq = chisquare(data2['decays_cesium'], norm.pdf(data2['decays_cesium'], mu_cesium, sigma_cesium))

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
        st.warning("Please upload both data1.csv and data2.csv files.")

if __name__ == "__main__":
    main()
