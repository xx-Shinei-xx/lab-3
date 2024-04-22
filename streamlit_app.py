import pandas as pd
import numpy as np
from scipy.stats import poisson, norm, chisquare
import matplotlib.pyplot as plt
data1 = pd.read_csv('data1.csv')  # Adjust the file path as needed
data2 = pd.read_csv('data2.csv')  # Adjust the file path as needed
# Fit Poisson distribution to data1
lambda_poisson_1 = np.mean(data1['values'])  # lambda parameter estimation from data1
poisson_expected_1 = poisson.pmf(data1['values'], mu=lambda_poisson_1)

# Perform chi-square goodness-of-fit test for data1
chi2_stat_1, p_val_1 = chisquare(f_obs=data1['counts'], f_exp=poisson_expected_1)
print("Chi-square statistic for data1:", chi2_stat_1)
print("p-value for data1:", p_val_1)
# Fit Poisson distribution to data2
lambda_poisson_2 = np.mean(data2['values'])  # lambda parameter estimation from data2
poisson_expected_2 = poisson.pmf(data2['values'], mu=lambda_poisson_2)

# Perform chi-square goodness-of-fit test for data2
chi2_stat_2, p_val_2 = chisquare(f_obs=data2['counts'], f_exp=poisson_expected_2)
print("Chi-square statistic for data2:", chi2_stat_2)
print("p-value for data2:", p_val_2)
# Fit Gaussian distribution to data1
mu_normal_1, std_normal_1 = norm.fit(data1['values'])  # mean and standard deviation estimation from data1
normal_expected_1 = norm.pdf(data1['values'], loc=mu_normal_1, scale=std_normal_1)

# Perform chi-square goodness-of-fit test for data1
chi2_stat_normal_1, p_val_normal_1 = chisquare(f_obs=data1['counts'], f_exp=normal_expected_1)
print("Chi-square statistic for data1 (Normal):", chi2_stat_normal_1)
print("p-value for data1 (Normal):", p_val_normal_1)
# Fit Gaussian distribution to data2
mu_normal_2, std_normal_2 = norm.fit(data2['values'])  # mean and standard deviation estimation from data2
normal_expected_2 = norm.pdf(data2['values'], loc=mu_normal_2, scale=std_normal_2)

# Perform chi-square goodness-of-fit test for data2
chi2_stat_normal_2, p_val_normal_2 = chisquare(f_obs=data2['counts'], f_exp=normal_expected_2)
print("Chi-square statistic for data2 (Normal):", chi2_stat_normal_2)
print("p-value for data2 (Normal):", p_val_normal_2)
