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
    poisson_lambda = np.mean(data)  # Use sample mean as Poisson lambda parameter
    expected_poisson = poisson.pmf(bin_centers, poisson_lambda) * len(data)

    # Compute chi-square goodness-of-fit statistic
    _, poisson_p_value = chisquare(counts, f_exp=expected_poisson)

    # Fit Gaussian distribution
    gaussian_params = norm.fit(data)

    return poisson_lambda, poisson_p_value, gaussian_params

def main():
    st.title("Distribution Fitting App")

    # Load data from CSV files
    data1 = pd.read_csv('data1.csv', header=None, names=['values'])
    data2 = pd.read_csv('data2.csv', header=None, names=['values'])

    st.header("Data 1")

    # Display histogram for Data 1
    fig1, ax1 = plt.subplots()
    ax1.hist(data1['values'], bins='auto', alpha=0.75, color='blue', edgecolor='black')
    ax1.set_title("Histogram for Data 1")
    st.pyplot(fig1)

    # Fit distributions and calculate goodness-of-fit for Data 1
    poisson_lambda1, poisson_p_value1, gaussian_params1 = fit_and_test(data1['values'])

    st.write(f"**Poisson Lambda (Data 1):** {poisson_lambda1:.4f}")
    st.write(f"**Poisson Chi-square p-value (Data 1):** {poisson_p_value1:.4f}")
    st.write("**Gaussian Fit Parameters (Data 1):**", gaussian_params1)

    st.header("Data 2")

    # Display histogram for Data 2
    fig2, ax2 = plt.subplots()
    ax2.hist(data2['values'], bins='auto', alpha=0.75, color='green', edgecolor='black')
    ax2.set_title("Histogram for Data 2")
    st.pyplot(fig2)

    # Fit distributions and calculate goodness-of-fit for Data 2
    poisson_lambda2, poisson_p_value2, gaussian_params2 = fit_and_test(data2['values'])

    st.write(f"**Poisson Lambda (Data 2):** {poisson_lambda2:.4f}")
    st.write(f"**Poisson Chi-square p-value (Data 2):** {poisson_p_value2:.4f}")
    st.write("**Gaussian Fit Parameters (Data 2):**", gaussian_params2)

if __name__ == "__main__":
    main()
