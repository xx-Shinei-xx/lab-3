import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import poisson, norm, chi2_contingency

# Load data from CSV file in the same directory
data1 = pd.read_csv('data1.csv')

# Define the Streamlit app
def main():
    st.title('Distribution Fitting and χ² Test')

    # Display a portion of the loaded data
    st.subheader('Sample Data')
    st.write(data1.head())

    # Buttons to choose distribution type
    distribution_type = st.radio("Select Distribution:", ('Poisson', 'Gaussian'))

    if distribution_type == 'Poisson':
        # Poisson distribution fitting
        poisson_lambda = st.slider('λ (Poisson parameter):', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        expected_values = [poisson.pmf(k, poisson_lambda) * len(data1) for k in range(len(data1))]
        observed_values = data1['measurements'].values
        _, p_value = chi2_contingency([observed_values, expected_values])

        st.subheader(f'Poisson Distribution (λ={poisson_lambda})')
        st.write(f'χ² Test p-value: {p_value:.4f}')
        st.bar_chart(data={'Observed': observed_values, 'Expected': expected_values})

    elif distribution_type == 'Gaussian':
        # Gaussian distribution fitting
        mean = st.slider('Mean:', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        std_dev = st.slider('Standard Deviation:', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        expected_values = [norm.pdf(x, mean, std_dev) * len(data1) for x in data1['measurements']]
        observed_values = data1['measurements'].values
        _, p_value = chi2_contingency([observed_values, expected_values])

        st.subheader(f'Gaussian Distribution (Mean={mean}, Std Dev={std_dev})')
        st.write(f'χ² Test p-value: {p_value:.4f}')
        st.bar_chart(data={'Observed': observed_values, 'Expected': expected_values})

# Run the app
if __name__ == '__main__':
    main()

