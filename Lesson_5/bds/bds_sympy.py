import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy as sc


#---------------------- Preface ------------------------------
# Import data
raw_data = pd.read_excel("./data/usd_2023-2024_1.xlsx", "USD")
usd_data = np.array(raw_data.iloc[0:161, 6].values)
plt.plot(usd_data)
plt.show()

# String define model
raw_model = "a0 + a1*t + a2*sin(a3*t) + a4*cos(a5*t)"
model = sp.parse_expr(raw_model)

# String define poly
raw_poly = "c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5"
poly = sp.parse_expr(raw_poly)

# Define main var
t = sp.var("t")

print('----------- model "in" --------------')
print(model)
print(poly)

# Do the LMS thing
poly_vals = np.polyfit(np.arange(0, usd_data.size), usd_data, 5)
poly_vals

# Do the DSB thing
series = sp.series(model, t).removeO()
poly_s = sp.series(poly, t).removeO()

print('----------- model "out" --------------')
print(series)
print(poly_s)

print('---------- model "coeffs" -------------')
model_coeffs = sp.poly(series, t).coeffs()
poly_coeffs = sp.poly(poly_s, t).coeffs()
for i, c in enumerate(model_coeffs):
    print(model_coeffs[i], poly_coeffs[i])


print('------- model "coeffs: a,c" -----------')
model_a = model.free_symbols
model_a.remove(t)
poly_c = poly.free_symbols
poly_c.remove(t)
print(model_a)
print(poly_c)


model_plain = model.copy()
for a in model_a:
    model_plain = model_plain.subs(a, 1)

if model_plain.has(1):
    model_plain = model_plain - 1

model_plain

series_plain = sp.series(model_plain, t).removeO()
series_plain

model_plain_coeffs = sp.poly(series_plain, t).coeffs()
model_plain_coeffs

print('--------- model discretes -----------')
discretes = []
for i in range(len(model_coeffs)):
    discretes.append(model_coeffs[i] / model_plain_coeffs[i])

for d in discretes:
    print(d)

print('--------- model balance -----------')
balance = []
for i, d in enumerate(discretes):
    balance.append(sp.Eq(d, poly_coeffs[i]))

for d in balance:
    print(d)


print('--------- model solution -----------')
model_a_list = sorted(model_a, key=lambda s: s.name)
solution = sp.nonlinsolve(balance, model_a_list)
if len(solution) > 1:
    solution = list(solution)[0]
print(model_a_list, solution)


#----------------- Substitution -----------------------------


# Substitute original model
for i, a in enumerate(model_a_list):
    model = model.subs(a, solution[i])
model.simplify()

# Substitute with calculated poly values
for i, c in enumerate(poly_coeffs):
    model = model.subs(c, poly_vals[i])
model

for i, c in enumerate(poly_coeffs):
    solution = solution.subs(c, poly_vals[i])
solution

model_y = np.ndarray(len(usd_data))
for i in np.arange(model_y.size):
    model_y[i] = model.subs(t, i)

plt.plot(usd_data)
plt.plot(model_y)
plt.show()


# ---------------- Numeric -----------------------------

# Do the numeric thing
model_r = sp.parse_expr(raw_model)
model_al = model_a_list.copy()
model_al.insert(0, t)
model_l = sp.lambdify(model_al, model_r)

sol = np.array(solution).astype(np.float64)

coeffs = sc.optimize.curve_fit(
    model_l, np.arange(usd_data.size), usd_data, p0=sol, maxfev=100000
)

model_yr = np.ndarray(len(usd_data))
for i in np.arange(usd_data.size):
    model_yr[i] = model_l(i, *coeffs[0])

plt.plot(usd_data)
plt.plot(model_yr)
plt.show()
