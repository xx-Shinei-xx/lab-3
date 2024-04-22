import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, chisquare
import matplotlib.pyplot as plt

# Load data from CSV file
@st.cache  # Cache data loading for improved performance
def load_data(file_path):
    return pd.read_csv(file_path)

def main():
    st.title('Distribution Fitting and Chi-Square Test')

    # Sidebar for file upload and parameters
    st.sidebar.header('Upload Data')
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        # Plot histogram
        st.header('Histogram of Measurements')
        plt.hist(data['measurement'], bins=20, alpha=0.75)
        plt.xlabel('Measurement')
        plt.ylabel('Frequency')
        st.pyplot()

        # Fit Poisson distribution
        mu_poisson = np.mean(data['measurement'])
        poisson_vals = poisson.pmf(data['measurement'], mu_poisson)

        # Plot Poisson fit
        st.header('Poisson Distribution Fit')
        plt.hist(data['measurement'], bins=20, density=True, alpha=0.75, label='Data')
        plt.plot(data['measurement'], poisson_vals, 'r-', lw=2, label='Poisson Fit')
        plt.xlabel('Measurement')
        plt.ylabel('Probability')
        st.pyplot()

        # Fit Gaussian (Normal) distribution
        mu_norm, sigma_norm = norm.fit(data['measurement'])
        norm_vals = norm.pdf(data['measurement'], mu_norm, sigma_norm)

        # Plot Normal fit
        st.header('Normal Distribution Fit')
        plt.hist(data['measurement'], bins=20, density=True, alpha=0.75, label='Data')
        plt.plot(data['measurement'], norm_vals, 'r-', lw=2, label='Normal Fit')
        plt.xlabel('Measurement')
        plt.ylabel('Probability')
        st.pyplot()

        # Perform Chi-Square Test
        observed_counts, _ = np.histogram(data['measurement'], bins=20)
        expected_counts_poisson = poisson_vals * len(data)
        expected_counts_norm = norm_vals * len(data)

        chi2_poisson, p_poisson = chisquare(observed_counts, expected_counts_poisson)
        chi2_norm, p_norm = chisquare(observed_counts, expected_counts_norm)

        # Display Chi-Square Test Results
        st.header('Chi-Square Test Results')
        st.write(f'Poisson Fit: chi2 = {chi2_poisson:.2f}, p-value = {p_poisson:.4f}')
        st.write(f'Normal Fit: chi2 = {chi2_norm:.2f}, p-value = {p_norm:.4f}')

        # Compare p-values to determine the best fit
        if p_poisson < p_norm:
            st.write('Poisson distribution provides a better fit to the data.')
        else:
            st.write('Normal distribution provides a better fit to the data.')

if __name__ == '__main__':
    main()

