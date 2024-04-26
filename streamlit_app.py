import numpy as np
from scipy.stats import poisson, norm
import matplotlib.pyplot as plt
import streamlit as st

# Load data
data = [...] # Load data from data1.csv

# Analysis functions
def poisson_analysis(data):
    # Perform Poisson distribution fit and chi-square test
    # Return results

def gaussian_analysis(data):
    # Perform Gaussian distribution fit and chi-square test
    # Return results

# Streamlit app
st.title("Data Analysis")

# Sidebar radio button to select distribution
distribution = st.sidebar.radio("Select Distribution", ("Poisson", "Gaussian"))

# Analysis section
if distribution == "Poisson":
    st.header("Poisson Distribution Analysis")
    poisson_results = poisson_analysis(data)
    # Display results using st.write(), st.pyplot(), etc.
else:
    st.header("Gaussian Distribution Analysis")
    gaussian_results = gaussian_analysis(data)
    # Display results using st.write(), st.pyplot(), etc.
