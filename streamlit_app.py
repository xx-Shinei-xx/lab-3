import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, chisquare

def fit_and_test(data):
    # Create a histogram to estimate observed frequencies
    counts, bin_edges = np.histogram(data, bins='auto')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit Poisson distribution
    poisson_lambda = np.mean(data) if len(data) > 0 else 0  # Use sample mean as Poisson lambda parameter, or 0 if data is empty
    expected_poisson = poisson.pmf(bin_centers, poisson_lambda) * len(data)

    # Compute chi-square goodness-of-fit statistic
    _, poisson_p_value = chisquare(counts, f_exp=expected_poisson)

    # Fit Gaussian distribution
    gaussian_params = norm.fit(data) if len(data) > 0 else (0, 0)  # Fit Gaussian if data is not empty, otherwise use (0, 0)

    return poisson_lambda, poisson_p_value, gaussian_params, bin_centers, expected_poisson

def main():
    st.title("Distribution Fitting App")

    # Load data from CSV files
    try:
        data1 = pd.read_csv('data1.csv', header=None, names=['values'])
        data2 = pd.read_csv('data2.csv', header=None, names=['values'])
    except FileNotFoundError:
        st.error("One or both CSV files ('data1.csv' and 'data2.csv') are missing.")
        return

    st.header("Data 1")

    # Display histogram for Data 1
    fig1, ax1 = plt.subplots()
    ax1.hist(data1['values'], bins='auto', alpha=0.75, color='blue', edgecolor='black', density=True)

    # Fit distributions and calculate goodness-of-fit for Data 1
    poisson_lambda1, poisson_p_value1, gaussian_params1, bin_centers1, expected_poisson1 = fit_and_test(data1['values'])
    ax1.plot(bin_centers1, expected_poisson1, linestyle='-', color='red', label='Poisson Fit')
    ax1.set_title("Histogram and Poisson Fit for Data 1")
    ax1.legend()
    st.pyplot(fig1)

    st.write(f"**Poisson Lambda (Data 1):** {poisson_lambda1:.4f}")
    st.write(f"**Poisson Chi-square p-value (Data 1):** {poisson_p_value1:.4f}")
    st.write("**Gaussian Fit Parameters (Data 1):**", gaussian_params1)

    st.header("Data 2")

    # Display histogram for Data 2
    fig2, ax2 = plt.subplots()
    ax2.hist(data2['values'], bins='auto', alpha=0.75, color='green', edgecolor='black', density=True)

    # Fit distributions and calculate goodness-of-fit for Data 2
    poisson_lambda2, poisson_p_value2, gaussian_params2, bin_centers2, expected_poisson2 = fit_and_test(data2['values'])
    ax2.plot(bin_centers2, expected_poisson2, linestyle='-', color='red', label='Poisson Fit')
    ax2.set_title("Histogram and Poisson Fit for Data 2")
    ax2.legend()
    st.pyplot(fig2)

    st.write(f"**Poisson Lambda (Data 2):** {poisson_lambda2:.4f}")
    st.write(f"**Poisson Chi-square p-value (Data 2):** {poisson_p_value2:.4f}")
    st.write("**Gaussian Fit Parameters (Data 2):**", gaussian_params2)

if __name__ == "__main__":
    main()
