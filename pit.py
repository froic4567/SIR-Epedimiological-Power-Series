import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================
# Step 1: Load Your Data (REPLACE WITH YOUR DATA)
# =============================================
# Time points: March 2020 (0) to December 2021 (21) → 22 months
months = np.arange(0, 22)  # 0 = March 2020, 1 = April 2020, ..., 21 = Dec 2021

# Replace with Cagayan de Oro's population for 2021
N = 700000  # Example: Use the 2021 population value

# Replace these arrays with your actual data!
I_actual = np.array([100, 200, 500, 1200, ..., 150])  # Infected (22 values)
R_actual = np.array([0, 50, 200, 600, ..., 1000])      # Recovered (22 values)

# Compute Susceptible (S = N - I - R)
S_actual = N - I_actual - R_actual

# Initial conditions (March 2020)
I0 = I_actual[0]
R0 = R_actual[0]
S0 = S_actual[0]

# =============================================
# Step 2: Power Series Coefficient Calculation
# =============================================
def compute_coefficients(beta, gamma, S0, I0, R0, max_order=15):
    s = [S0]
    i = [I0]
    r = [R0]
    
    for n in range(max_order):
        sum_si = sum(s[k] * i[n - k] for k in range(n + 1))
        s_new = -beta / (n + 1) * sum_si
        i_new = (beta / (n + 1) * sum_si) - (gamma * i[n]) / (n + 1)
        r_new = (gamma * i[n]) / (n + 1)
        
        s.append(s_new)
        i.append(i_new)
        r.append(r_new)
    
    return s, i, r

# =============================================
# Step 3: Define the Power Series Model
# =============================================
def power_series_model(t, beta, gamma, max_order=15):
    s_coeff, i_coeff, r_coeff = compute_coefficients(beta, gamma, S0, I0, R0, max_order)
    S_pred = sum(s_coeff[n] * t**n for n in range(len(s_coeff)))
    I_pred = sum(i_coeff[n] * t**n for n in range(len(i_coeff)))
    R_pred = sum(r_coeff[n] * t**n for n in range(len(r_coeff)))
    return S_pred, I_pred, R_pred

# =============================================
# Step 4: Parameter Estimation (Beta and Gamma)
# =============================================
def fit_function(params, t):
    beta, gamma = params
    S_pred, I_pred, R_pred = power_series_model(t, beta, gamma)
    return np.concatenate([I_pred, R_pred])

observed_data = np.concatenate([I_actual, R_actual])
initial_guess = [0.3, 0.1]
params, _ = curve_fit(lambda t, beta, gamma: fit_function((beta, gamma), t), 
                      months, observed_data, p0=initial_guess, bounds=(0, [5, 5]))
beta_fit, gamma_fit = params
print(f"Fitted: beta = {beta_fit:.3f}/month, gamma = {gamma_fit:.3f}/month")

# =============================================
# Step 5: Generate Predictions
# =============================================
S_pred, I_pred, R_pred = [], [], []
for t in months:
    S, I, R = power_series_model(t, beta_fit, gamma_fit)
    S_pred.append(S)
    I_pred.append(I)
    R_pred.append(R)

# Ensure S + I + R = N
S_pred = np.clip(S_pred, 0, N)
I_pred = np.clip(I_pred, 0, N)
R_pred = np.clip(R_pred, 0, N - S_pred - I_pred)

# =============================================
# Step 6: Visualize Results
# =============================================
plt.figure(figsize=(12, 8))

# Plot Infected
plt.subplot(2, 2, 1)
plt.plot(months, I_actual, 'ro-', label="Actual Infected")
plt.plot(months, I_pred, 'b--', label="Predicted Infected")
plt.xlabel("Months (0 = Mar 2020)")
plt.ylabel("Infected")
plt.title("Infected: Model vs. Data")
plt.legend()

# Plot Recovered
plt.subplot(2, 2, 2)
plt.plot(months, R_actual, 'co-', label="Actual Recovered")
plt.plot(months, R_pred, 'y--', label="Predicted Recovered")
plt.xlabel("Months (0 = Mar 2020)")
plt.ylabel("Recovered")
plt.title("Recovered: Model vs. Data")
plt.legend()

# Plot Susceptible
plt.subplot(2, 2, 3)
plt.plot(months, S_actual, 'go-', label="Actual Susceptible")
plt.plot(months, S_pred, 'm--', label="Predicted Susceptible")
plt.xlabel("Months (0 = Mar 2020)")
plt.ylabel("Susceptible")
plt.title("Susceptible: Model vs. Data")
plt.legend()

plt.tight_layout()
plt.show()