#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import stats


# In[2]:


data = [1.13, 1.55, 1.43, 0.92, 1.25, 1.36, 1.32, 0.85, 1.07, 1.48, 1.20, 1.33, 1.18, 1.22, 1.29]


# In[3]:


#Task a: Build 99% Confidence Interval Using Sample Standard Deviation

n = len(data)

sample_mean = np.mean(data)

sample_std = np.std(data, ddof=1)

t_value = stats.t.ppf(0.995, df=n-1) 

margin_of_error = t_value * (sample_std / np.sqrt(n))

ci_lower = sample_mean - margin_of_error

ci_upper = sample_mean + margin_of_error

print("Task a: Confidence Interval Using Sample Standard Deviation")
print(f"Sample Mean: {sample_mean:.3f}")
print(f"Sample Standard Deviation: {sample_std:.3f}")
print(f"T-value: {t_value:.3f}")
print(f"Margin of Error: {margin_of_error:.3f}")
print(f"99% Confidence Interval: ({ci_lower:.3f}, {ci_upper:.3f})")
print()


# In[4]:


#Task b: Build 99% Confidence Interval Using Known Population Standard Deviation
population_std = 0.2

z_value = stats.norm.ppf(0.995)

margin_of_error_known_pop = z_value * (population_std / np.sqrt(n))

ci_lower_known_pop = sample_mean - margin_of_error_known_pop

ci_upper_known_pop = sample_mean + margin_of_error_known_pop

print("Task b: Confidence Interval Using Known Population Standard Deviation")
print(f"Z-value: {z_value:.3f}")
print(f"Margin of Error (Known Population SD): {margin_of_error_known_pop:.3f}")
print(f"99% Confidence Interval: ({ci_lower_known_pop:.3f}, {ci_upper_known_pop:.3f})")


# In[ ]:




