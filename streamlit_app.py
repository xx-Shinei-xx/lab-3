import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, chisquare

# Load data from CSV files
data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')

# Fit Poisson distribution to data1
lambda_poisson_1 = np.mean(data1['measurement'])
poisson_expected_1 = poisson.pmf(data1['measurement'], mu=lambda_poisson_1)

# Perform chi-square goodness-of-fit test for data1
chi2_stat_1, p_val_1 = chisquare(f_obs=data1['counts'], f_exp=poisson_expected_1)
print("Chi-square statistic for data1:", chi2_stat_1)
print("p-value for data1:", p_val_1)

# Fit Poisson distribution to data2
lambda_poisson_2 = np.mean(data2['measurement'])
poisson_expected_2 = poisson.pmf(data2['measurement'], mu=lambda_poisson_2)

# Perform chi-square goodness-of-fit test for data2
chi2_stat_2, p_val_2 = chisquare(f_obs=data2['counts'], f_exp=poisson_expected_2)
print("Chi-square statistic for data2:", chi2_stat_2)
print("p-value for data2:", p_val_2)
