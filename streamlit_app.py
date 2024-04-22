import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, chisquare
import matplotlib.pyplot as plt

data1 = pd.read_csv('data1.csv')
print(data1.head())
# Fit Poisson Distribution
poisson_params = poisson.fit(data1)

# Fit Gaussian (Normal) Distribution
norm_params = norm.fit(data1)
plt.hist(data1, bins=20, density=True, alpha=0.6, color='g', label='Data')

# Plot Fitted Poisson Distribution
x_poisson = np.arange(0, max(data1)+1)
plt.plot(x_poisson, poisson.pmf(x_poisson, *poisson_params), 'r-', label='Poisson')

# Plot Fitted Gaussian Distribution
x_norm = np.linspace(min(data1), max(data1), 100)
plt.plot(x_norm, norm.pdf(x_norm, *norm_params), 'b-', label='Normal')

plt.title('Histogram of Data with Fitted Distributions')
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
# Calculate observed and expected frequencies for Poisson
observed_poisson, _ = np.histogram(data1, bins=np.arange(-0.5, max(data1)+1.5, 1))
expected_poisson = poisson.pmf(np.arange(0, max(data1)+1), *poisson_params) * len(data1)

# Calculate observed and expected frequencies for Normal
observed_normal, _ = np.histogram(data1, bins=20, density=True)
expected_normal = norm.pdf(np.linspace(min(data1), max(data1), 20), *norm_params) * len(data1)

# Perform Chi-Square Test for Poisson
chi2_poisson, p_poisson = chisquare(observed_poisson, expected_poisson)

# Perform Chi-Square Test for Normal
chi2_normal, p_normal = chisquare(observed_normal, expected_normal)

print(f'Chi-Square Test for Poisson: chi2 = {chi2_poisson}, p-value = {p_poisson}')
print(f'Chi-Square Test for Normal: chi2 = {chi2_normal}, p-value = {p_normal}')
