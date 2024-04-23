import pandas as pd
import numpy as np
from scipy.stats import poisson, norm
from scipy.stats import chi2

# Read data from CSV files
data1 = pd.read_csv('data1.csv', header=None, names=['values'])
data2 = pd.read_csv('data2.csv', header=None, names=['values'])

# Poisson Fit for data1
data1_mean = data1['values'].mean()
poisson_params = poisson.fit(data1['values'].values, floc=0)
poisson_pdf = poisson.pmf(data1['values'], poisson_params[:-2])
poisson_chi2, poisson_p_value1 = chi2.cdft(poisson_pdf, data1['values'])

# Gaussian Fit for data1
gaussian_params = norm.fit(data1['values'].values)
gaussian_pdf = norm.pdf(data1['values'], gaussian_params[0], gaussian_params[1])
gaussian_chi2, gaussian_p_value1 = chi2.cdft(gaussian_pdf, data1['values'])

# Poisson Fit for data2
data2_mean = data2['values'].mean()
poisson_params = poisson.fit(data2['values'].values, floc=0)
poisson_pdf = poisson.pmf(data2['values'], poisson_params[:-2])
poisson_chi2, poisson_p_value2 = chi2.cdft(poisson_pdf, data2['values'])

# Gaussian Fit for data2
gaussian_params = norm.fit(data2['values'].values)
gaussian_pdf = norm.pdf(data2['values'], gaussian_params[0], gaussian_params[1])
gaussian_chi2, gaussian_p_value2 = chi2.cdft(gaussian_pdf, data2['values'])

print(f"Poisson fit for data1: mean = {data1_mean:.2f}, chi2 = {poisson_chi2:.2f}, p-value = {poisson_p_value1:.4f}")
print(f"Gaussian fit for data1: mean = {gaussian_params[0]:.2f}, std = {gaussian_params[1]:.2f}, chi2 = {gaussian_chi2:.2f}, p-value = {gaussian_p_value1:.4f}")

print(f"\nPoisson fit for data2: mean = {data2_mean:.2f}, chi2 = {poisson_chi2:.2f}, p-value = {poisson_p_value2:.4f}")
print(f"Gaussian fit for data2: mean = {gaussian_params[0]:.2f}, std = {gaussian_params[1]:.2f}, chi2 = {gaussian_chi2:.2f}, p-value = {gaussian_p_value2:.4f}")


#if __name__ == "__main__":
#    main()
