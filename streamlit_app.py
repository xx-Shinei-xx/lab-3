import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, chisquare

def fit_and_test(data):
    # Fit Poisson distribution
    poisson_params = poisson.fit(data, floc=0)
    poisson_pdf = poisson.pmf(data, *poisson_params[:-2])
    _, poisson_p_value = chisquare(data, poisson_pdf)

    # Fit Gaussian distribution
    gaussian_params = norm.fit(data)
    gaussian_pdf = norm.pdf(data, *gaussian_params)
    _, gaussian_p_value = chisquare(data, gaussian_pdf)
    
    return poisson_params, poisson_p_value, gaussian_params, gaussian_p_value

def main():
    st.title("Distribution Fitting App")

    # Load data from CSV files
    data1 = pd.read_csv('data1.csv', header=None, names=['values'])
    data2 = pd.read_csv('data2.csv', header=None, names=['values'])

    st.header("Data 1")

    # Display histogram for Data 1
    fig1, ax1 = plt.subplots()
    ax1.hist(data1['values'], bins=20, alpha=0.75, color='blue', edgecolor='black')
    ax1.set_title("Histogram for Data 1")
    st.pyplot(fig1)

    # Fit distributions and calculate goodness-of-fit for Data 1
    poisson_params1, poisson_p_value1, gaussian_params1, gaussian_p_value1 = fit_and_test(data1['values'])

    st.write("**Poisson Fit Parameters:**", poisson_params1)
    st.write(f"**Poisson Chi-square p-value:** {poisson_p_value1:.4f}")
    st.write("**Gaussian Fit Parameters:**", gaussian_params1)
    st.write(f"**Gaussian Chi-square p-value:** {gaussian_p_value1:.4f}")

    st.header("Data 2")

    # Display histogram for Data 2
    fig2, ax2 = plt.subplots()
    ax2.hist(data2['values'], bins=20, alpha=0.75, color='green', edgecolor='black')
    ax2.set_title("Histogram for Data 2")
    st.pyplot(fig2)

    # Fit distributions and calculate goodness-of-fit for Data 2
    poisson_params2, poisson_p_value2, gaussian_params2, gaussian_p_value2 = fit_and_test(data2['values'])

    st.write("**Poisson Fit Parameters:**", poisson_params2)
    st.write(f"**Poisson Chi-square p-value:** {poisson_p_value2:.4f}")
    st.write("**Gaussian Fit Parameters:**", gaussian_params2)
    st.write(f"**Gaussian Chi-square p-value:** {gaussian_p_value2:.4f}")

if __name__ == "__main__":
    main()
