import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson, norm

# Load data from CSV file
   # Cache the data for improved performance
def load_data(filename):
    return pd.read_csv(filename)

# Function to generate plot based on distribution type and parameters
def plot_distribution(data, dist_type, param_value):
    if dist_type == 'Poisson':
        dist = poisson(mu=param_value)
        title = f'Poisson Distribution (λ={param_value})'
    elif dist_type == 'Gaussian':
        dist = norm(loc=param_value)
        title = f'Gaussian Distribution (μ={param_value})'
    else:
        st.error('Invalid distribution type selected.')
        return

    # Calculate PDF values for a range of x values
    x = np.linspace(min(data), max(data), 100)
    pdf_values = dist.pdf(x)  # Calculate PDF values for the selected distribution

    # Create a plotly figure
    fig = go.Figure()

    # Add histogram trace
    fig.add_trace(go.Histogram(x=data, histnorm='density', name='Data Histogram'))

    # Add PDF trace for the selected distribution
    fig.add_trace(go.Scatter(x=x, y=pdf_values, mode='lines', name=f'{dist_type} PDF'))

    # Update figure layout
    fig.update_layout(
        title=title,
        xaxis_title='Value',
        yaxis_title='Density',
        showlegend=True
    )

    # Display the plotly figure
    st.plotly_chart(fig)

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
    else:
        st.error('Invalid distribution type selected.')
        return

    # Parameter slider
    parameter_value = st.slider(f'Select {parameter_label}:', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    # Plot distribution based on selection
    plot_distribution(values, distribution_type, parameter_value)

if __name__ == '__main__':
    main()
