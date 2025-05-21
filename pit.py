import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================
# Step 1: Load Your Data (Replace with your data)
# =============================================
# Example data format (monthly from Jan 2021 to Dec 2022, 24 months)
# Replace these arrays with your actual data!
months = np.arange(0, 24)  # Time points (0 = Jan 2021, 1 = Feb 2021, ..., 23 = Dec 2022)
S_actual = np.array([999000, 998500, 997800, ...])  # Susceptible (hypothetical data)
I_actual = np.array([1000, 1500, 2200, ...])         # Infected
R_actual = np.array([0, 500, 1200, ...])             # Recovered

# Total population (assumed constant)
N = S_actual[0] + I_actual[0] + R_actual[0]

# =============================================
# Step 2: Power Series Coefficient Calculation
# =============================================
def compute_coefficients(beta, gamma, S0, I0, R0, max_order=15):
    """
    Compute power series coefficients for S(t), I(t), R(t).
    """
    s = [S0]
    i = [I0]
    r = [R0]
    
    for n in range(max_order):
        # Compute sums for recurrence relations
        sum_si = sum(s[k] * i[n - k] for k in range(n + 1))
        
        # Recurrence relations (Equations 6-7 from the paper)
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
    """
    Predict S(t), I(t), R(t) using power series up to `max_order`.
    """
    S0 = S_actual[0]
    I0 = I_actual[0]
    R0 = R_actual[0]
    
    s_coeff, i_coeff, r_coeff = compute_coefficients(beta, gamma, S0, I0, R0, max_order)
    
    # Evaluate the series at time t (month)
    S_pred = sum(s_coeff[n] * (t)**n for n in range(len(s_coeff)))
    I_pred = sum(i_coeff[n] * (t)**n for n in range(len(i_coeff)))
    R_pred = sum(r_coeff[n] * (t)**n for n in range(len(r_coeff)))
    
    return S_pred, I_pred, R_pred

# =============================================
# Step 4: Parameter Estimation (Beta and Gamma)
# =============================================
# Define a function to minimize (fit to infected data)
def fit_function(t, beta, gamma):
    _, I_pred, _ = power_series_model(t, beta, gamma)
    return I_pred

# Fit beta and gamma to the infected data
initial_guess = [0.3, 0.1]  # Initial guess for beta and gamma (per month)
params, _ = curve_fit(fit_function, months, I_actual, p0=initial_guess, bounds=(0, [5, 5]))
beta_fit, gamma_fit = params
print(f"Fitted parameters: beta = {beta_fit:.3f}/month, gamma = {gamma_fit:.3f}/month")

# =============================================
# Step 5: Generate Predictions
# =============================================
# Compute predictions for all months
S_pred = []
I_pred = []
R_pred = []
for t in months:
    S, I, R = power_series_model(t, beta_fit, gamma_fit)
    S_pred.append(S)
    I_pred.append(I)
    R_pred.append(R)

# =============================================
# Step 6: Visualize Results
# =============================================
plt.figure(figsize=(12, 6))

# Plot Infected
plt.subplot(1, 2, 1)
plt.plot(months, I_actual, 'ro-', label="Actual Infected")
plt.plot(months, I_pred, 'b--', label="Power Series Prediction")
plt.xlabel("Months (0 = Jan 2021)")
plt.ylabel("Number of Infected People")
plt.title("Infected: Model vs. Data")
plt.legend()

# Plot Susceptible and Recovered
plt.subplot(1, 2, 2)
plt.plot(months, S_actual, 'go-', label="Actual Susceptible")
plt.plot(months, S_pred, 'm--', label="Predicted Susceptible")
plt.plot(months, R_actual, 'co-', label="Actual Recovered")
plt.plot(months, R_pred, 'y--', label="Predicted Recovered")
plt.xlabel("Months (0 = Jan 2021)")
plt.ylabel("Population")
plt.title("Susceptible and Recovered")
plt.legend()

plt.tight_layout()
plt.show()