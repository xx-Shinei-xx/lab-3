import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, chi2
import matplotlib.pyplot as plt


# Sample data (replace with your actual data)
data1 = np.array([10, 8, 12, 14, 9, 11, 13, 7, 10, 12])
data2 = np.array([25, 27, 30, 28, 26, 24, 29, 31, 27, 28])

# Fit Poisson and Gaussian distributions to the data
mu_poisson_data1 = np.mean(data1)
poisson_fit_data1 = poisson(mu_poisson_data1)

mean_data1, std_data1 = np.mean(data1), np.std(data1)
normal_fit_data1 = norm(mean_data1, std_data1)

mu_poisson_data2 = np.mean(data2)
poisson_fit_data2 = poisson(mu_poisson_data2)

mean_data2, std_data2 = np.mean(data2), np.std(data2)
normal_fit_data2 = norm(mean_data2, std_data2)

# Define chi-square test function
def chi_square_test(observed, expected):
    # Ensure observed and expected have the same length
    min_len = min(len(observed), len(expected))
    observed = observed[:min_len]
    expected = expected[:min_len]
    
    chi2_statistic = np.sum((observed - expected)**2 / expected)
    df = min_len - 1
    p_value = 1 - chi2.cdf(chi2_statistic, df)
    return chi2_statistic, p_value

# Perform chi-square tests for data1
observed_counts1, _ = np.histogram(data1, bins=np.arange(0, np.max(data1)+2, 1))
expected_counts_poisson1 = np.array([poisson_fit_data1.pmf(k) * len(data1) for k in range(np.max(data1)+1)])
chi2_statistic_poisson1, p_value_poisson1 = chi_square_test(observed_counts1, expected_counts_poisson1)

expected_counts_normal1 = np.array([normal_fit_data1.pdf(x) * len(data1) for x in np.arange(np.min(data1), np.max(data1)+1, 1)])
chi2_statistic_normal1, p_value_normal1 = chi_square_test(observed_counts1, expected_counts_normal1)

# Perform chi-square tests for data2
observed_counts2, _ = np.histogram(data2, bins=np.arange(0, np.max(data2)+2, 1))
expected_counts_poisson2 = np.array([poisson_fit_data2.pmf(k) * len(data2) for k in range(np.max(data2)+1)])
chi2_statistic_poisson2, p_value_poisson2 = chi_square_test(observed_counts2, expected_counts_poisson2)

expected_counts_normal2 = np.array([normal_fit_data2.pdf(x) * len(data2) for x in np.arange(np.min(data2), np.max(data2)+1, 1)])
chi2_statistic_normal2, p_value_normal2 = chi_square_test(observed_counts2, expected_counts_normal2)

# Print results
print("Chi-square test results for data1:")
print("Poisson distribution - Chi2 =", chi2_statistic_poisson1, "  p-value =", p_value_poisson1)
print("Gaussian distribution - Chi2 =", chi2_statistic_normal1, "  p-value =", p_value_normal1)

print("\nChi-square test results for data2:")
print("Poisson distribution - Chi2 =", chi2_statistic_poisson2, "  p-value =", p_value_poisson2)
print("Gaussian distribution - Chi2 =", chi2_statistic_normal2, "  p-value =", p_value_normal2)
