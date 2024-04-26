import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm

# Load data from CSV file
  # Cache the data for improved performance
def load_data(filename):
    return pd.read_csv(filename)

# Function to generate plot based on distribution type and parameters
def plot_distribution(data, dist_type, param_value):
    plt.figure(figsize=(8, 6))
    
    if dist_type == 'Poisson':
        dist = poisson(mu=param_value)
        title = f'Poisson Distribution (λ={param_value})'
    elif dist_type == 'Gaussian':
        dist = norm(loc=param_value)
        title = f'Gaussian Distribution (μ={param_value})'

    # Plot histogram of the data
    counts, bins, _ = plt.hist(data, bins=30, alpha=0.7, density=True, label='Data Histogram')

    # Plot the probability density function (PDF) of the selected distribution
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, dist.pdf(x), 'r-', lw=2, label=f'{dist_type} PDF')

    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    st.pyplot()

def main():
    st.title('Distribution Selector App')

    # Load data
    filename = 'data1.csv'
    data = load_data(filename)

    # Display data in a table
    st.write('### Data from CSV:')
    st.write(data)

    # Extract values from DataFrame column
    values = data['Decaimiento solo con el aire']

    # Distribution selector
    distribution_type = st.selectbox('Select Distribution Type:', ['Poisson', 'Gaussian'])

    if distribution_type == 'Poisson':
        parameter_label = 'λ (Lambda, Mean)'
    elif distribution_type == 'Gaussian':
        parameter_label = 'μ (Mean)'

    # Parameter slider
    parameter_value = st.slider(f'Select {parameter_label}:', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    # Plot distribution based on selection
    plot_distribution(values, distribution_type, parameter_value)

if __name__ == '__main__':
    main()
