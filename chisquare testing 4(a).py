#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import scipy.stats as stats


# In[3]:


#given data
data={
    "Satisfaction": ["Very Satisfied", "Satisfied", "Neutral", "Unsatisfied", "Very Unsatisfied"],
    "Smart Thermostat": [50, 80, 60, 30, 20],
    "Smart Light": [70, 100, 90, 50, 50],
    "Total": [120, 180, 150, 80, 70]
}


# In[4]:


data


# In[8]:


df=pd.DataFrame(data)


# In[9]:


df


# # Task 1: State the Hypotheses
# H0 (Null Hypothesis): There is no significant association between device type (Smart Thermostats vs. Smart Lights) and customer satisfaction level.
# 
# 
# HA (Alternative Hypothesis): There is a significant association between device type and customer satisfaction level.

# In[11]:


print("Task 1: Stating the Hypotheses")
print("H0 (Null Hypothesis): There is no significant association between device type and customer satisfaction level.")
print("HA (Alternative Hypothesis): There is a significant association between device type and customer satisfaction level.")
print()


# # Task 2: Compute the Chi-Square Statistic

# In[16]:


# We need to calculate the observed and expected counts and then compute the chi-square statistic.
# Observed counts
observed_counts = np.array([df["Smart Thermostat"], df["Smart Light"]])


# In[17]:


# Calculate the Chi-Square statistic and the p-value
chi2_stat, p_value, dof, expected_counts = chi2_contingency(observed_counts, correction=False)


# In[18]:


print("Task 2: Compute the Chi-Square Statistic")
print(f"Chi-Square Statistic: {chi2_stat:.3f}")
print(f"P-Value: {p_value:.3f}")
print(f"Degrees of Freedom: {dof}")
print()


# # Task 3: Determine the Critical Value
# 

# In[19]:


# Using the significance level (alpha) of 0.05 and the degrees of freedom, we determine the critical value.
# Degrees of freedom (df) = (number of rows - 1) * (number of columns - 1)
critical_value = stats.chi2.ppf(0.95, dof)
print("Task 3: Determine the Critical Value")
print(f"Critical Value (with alpha = 0.05): {critical_value:.3f}")
print()


# # Task 4: Make a Decision

# In[20]:


# Compare the Chi-Square statistic with the critical value to decide whether to reject the null hypothesis.
reject_null = chi2_stat > critical_value
print("Task 4: Make a Decision")
if reject_null:
    print("Decision: Reject the Null Hypothesis. There is a significant association between device type and customer satisfaction.")
else:
    print("Decision: Fail to reject the Null Hypothesis. There is no significant association between device type and customer satisfaction.")


# # Analysis Summary
# 

# In[21]:


# Hypotheses:
## Null Hypothesis (H0): There no significant association between device type and customer satisfaction.
## Alternative Hypothesis (HA): There a significant association between device type and customer satisfaction.

    # Chi-Square Statistic: 12.011

    # Degrees of Freedom: 4

    # Critical Value: 9.488 (with alpha = 0.05)

    # Decision: Since the Chi-Square statistic (12.011) is greater than the critical value (9.488), we reject the null hypothesis. This indicates there's a significant association between device type and customer satisfaction level


# In[ ]:




