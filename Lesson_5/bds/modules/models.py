import numpy as np

# Configuration
# Experimental measurement quantity
n = 10000
# Experiments count
m = 100
# Abnormals configuration
abnormal_coeff = 3
abnormal_dens = 10
abnormal_count = int((n * abnormal_dens) / 100)
# Gauss config
mu = 0
sigma = 5


# Ideal trend simulation f(x) = 0.0000005 * x^2
def exponential_plain():
    model = np.zeros((n))
    for i in range(n):
        model[i] = 0.0000005 * i * i
    return model


# Generate normal errors using gauss distribution
def exponential_normal():
    model = exponential_plain()
    errors = np.random.normal(mu, sigma, n)
    for i in range(n):
        model[i] += errors[i]
    return model


# Generate abnormal errors
def exponential_abnormal():
    model = exponential_normal()
    abnormal_pos = np.zeros((n))

    # Fill in positions using normal distribution
    for i in range(n):
        abnormal_pos[i] = np.ceil(np.random.randint(1, n))

    # Fill in abnormals
    abnormals = np.random.normal(mu, sigma * abnormal_coeff, abnormal_count)
    for i in range(abnormal_count):
        model[int(abnormal_pos[i])] += abnormals[i]
    return model
