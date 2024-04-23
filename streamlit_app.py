import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import poisson, norm, chi2_contingency
import matplotlib.pyplot as plt

# Load data from CSV file in the same directory
data1 = pd.read_csv('data1.csv')

# Check the column names of data1
st.write("Column Names:", data1.columns)

# Identify the correct column name for measurements
measurements_column = "decaimiento solo con el aire"  # Adjust based on the actual column name

# Extract measurements from the DataFrame
measurements = data1[measurements_column]

# Define the Streamlit app
def main():
    st.title('Distribution Fitting and χ² Test')

    # Display a portion of the loaded data
    st.subheader('Sample Data')
    st.write(data1.head())

    # Display histogram of measurements
    st.subheader('Histogram of Measurements')
    plt.hist(measurements, bins=20, color='skyblue', edgecolor='black')
    st.pyplot()

    # Buttons to choose distribution type
    distribution_type = st.radio("Select Distribution:", ('Poisson', 'Gaussian'))

    if distribution_type == 'Poisson':
        # Poisson distribution fitting
        poisson_lambda = st.slider('λ (Poisson parameter):', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        expected_values = [poisson.pmf(k, poisson_lambda) * len(measurements) for k in range(len(measurements))]
        observed_counts, _ = np.histogram(measurements, bins=len(expected_values))
        _, p_value, _, _ = chi2_contingency([observed_counts, expected_values])

        st.subheader(f'Poisson Distribution (λ={poisson_lambda})')
        st.write(f'χ² Test p-value: {p_value:.4f}')

    elif distribution_type == 'Gaussian':
        # Gaussian distribution fitting
        mean = st.slider('Mean:', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        std_dev = st.slider('Standard Deviation:', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        expected_values = [norm.pdf(x, mean, std_dev) * len(measurements) for x in measurements]
        observed_counts, _ = np.histogram(measurements, bins=len(expected_values))
        _, p_value, _, _ = chi2_contingency([observed_counts, expected_values])

        st.subheader(f'Gaussian Distribution (Mean={mean}, Std Dev={std_dev})')
        st.write(f'χ² Test p-value: {p_value:.4f}')

# Run the app
if __name__ == '__main__':
    main()
