import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, chi2_contingency

# Function to calculate statistics and plot distributions
def analyze_data(data):
    # Convert values to numeric
    data['values'] = pd.to_numeric(data['values'], errors='coerce')
    data.dropna(subset=['values'], inplace=True)

    # Calculate mean and standard deviation
    mean = data['values'].mean()
    std = data['values'].std()

    # Create histogram
    counts, bin_edges = np.histogram(data['values'], density=True)

    # Fit Poisson and Gaussian distributions
    poisson_dist = poisson(mu=mean)
    gaussian_dist = norm(loc=mean, scale=std)

    # Calculate expected frequencies
    expected_poisson = poisson_dist.pmf(bin_edges[:-1])
    expected_gaussian = gaussian_dist.pdf(np.linspace(data['values'].min(), data['values'].max(), len(bin_edges) - 1))

    # Perform chi-square test
    chi2_poisson, p_value_poisson = chi2_contingency([counts, expected_poisson], correction=False)
    chi2_gaussian, p_value_gaussian = chi2_contingency([counts, expected_gaussian], correction=False)

    # Plot histograms and distributions
    plt.figure(figsize=(10, 6))
    plt.hist(data['values'], bins='auto', density=True, alpha=0.75, label='Data')
    plt.plot(np.linspace(data['values'].min(), data['values'].max(), len(bin_edges) - 1), poisson_dist.pmf(np.linspace(data['values'].min(), data['values'].max(), len(bin_edges) - 1)), 'r-', lw=2, label='Poisson')
    plt.plot(np.linspace(data['values'].min(), data['values'].max(), len(bin_edges) - 1), gaussian_dist.pdf(np.linspace(data['values'].min(), data['values'].max(), len(bin_edges) - 1)), 'g-', lw=2, label='Gaussian')
    plt.title('Histogram and Fitted Distributions')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    st.pyplot()

    # Display statistics
    st.write("### Statistical Analysis Results:")
    st.write(f"Poisson distribution chi-square statistic: {chi2_poisson:.2f}, p-value: {p_value_poisson:.4f}")
    st.write(f"Gaussian distribution chi-square statistic: {chi2_gaussian:.2f}, p-value: {p_value_gaussian:.4f}")

# Streamlit app
def main():
    st.title("Data Analysis with Histograms and Distributions")

    # Upload CSV files
    uploaded_file1 = st.file_uploader("Upload CSV file for Data 1:", type=["csv"])
    uploaded_file2 = st.file_uploader("Upload CSV file for Data 2:", type=["csv"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.write("### Data 1:")
        data1 = pd.read_csv(uploaded_file1, header=None, names=['values'])
        st.write(data1.head())

        st.write("### Data 2:")
        data2 = pd.read_csv(uploaded_file2, header=None, names=['values'])
        st.write(data2.head())

        st.write("### Analysis Results for Data 1:")
        analyze_data(data1)

        st.write("### Analysis Results for Data 2:")
        analyze_data(data2)

if __name__ == "__main__":
    main()
