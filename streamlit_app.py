import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, chisquare

# Load the CSV data from GitHub
url = 'https://github.com/xx-Shinei-xx/lab-3/blob/main/data1.csv'
df = pd.read_csv(url)

# Sidebar for selecting distribution
st.sidebar.title('Select Distribution')
distribution = st.sidebar.radio('Distribution', ('Poisson', 'Gaussian'))

# Main content
st.title('Distribution Fitting App')

# Show the dataset
st.subheader('Dataset')
st.write(df)

# Distribution fitting
if distribution == 'Poisson':
    st.subheader('Poisson Distribution')

    # Poisson parameter input
    mean_lambda = st.slider('Mean (λ)', min_value=0.1, max_value=30.0, value=5.0, step=0.1)

    # Generate Poisson distribution
    expected_counts = [poisson.pmf(k, mean_lambda) * len(df) for k in range(len(df))]
    
    # Chi-square test
    observed_counts = df['counts'].tolist()
    chi2_stat, p_val = chisquare(observed_counts, f_exp=expected_counts)
    
    st.write(f'Chi-square statistic: {chi2_stat}')
    st.write(f'p-value: {p_val}')
    
elif distribution == 'Gaussian':
    st.subheader('Gaussian Distribution')

    # Gaussian parameter input
    mean = st.slider('Mean', min_value=df['counts'].min(), max_value=df['counts'].max(), value=df['counts'].mean())
    std_dev = st.slider('Standard Deviation', min_value=0.1, max_value=30.0, value=5.0, step=0.1)

    # Generate Gaussian distribution
    x_values = np.linspace(df['counts'].min(), df['counts'].max(), len(df))
    expected_counts = [norm.pdf(x, mean, std_dev) * len(df) for x in x_values]
    
    # Chi-square test
    observed_counts = df['counts'].tolist()
    chi2_stat, p_val = chisquare(observed_counts, f_exp=expected_counts)
    
    st.write(f'Chi-square statistic: {chi2_stat}')
    st.write(f'p-value: {p_val}')


if __name__ == "__main__":
    main()
