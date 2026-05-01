# Alpha-Beta Filter: Why Dynamic and Stochastic Accuracy Trade Off

**The alpha-beta filter cannot achieve high accuracy for both noise suppression and fast-changing signal tracking simultaneously due to a fundamental mathematical inverse relationship.** When filter parameters α and β are set low for noise reduction (stochastic accuracy), the filter's lag error increases for dynamic signals. Conversely, high parameters that track dynamics well allow more noise through. This trade-off, inherent to all fixed-gain recursive filters, explains why your lecture described "dynamic and stochastic accuracy are relatively low"—any compromise between the two extremes yields moderate performance in both.

---

## The alpha-beta filter fundamentals

The **alpha-beta filter** (also called α-β, f-g, or g-h filter) is a simplified recursive state estimator that emerged from Cold War-era radar tracking systems in the 1950s. J. Sklansky at RCA Laboratories first formalized it in 1957 for track-while-scan radar, and Benedict-Bordner refined it in 1962 for steady-state performance optimization.

The filter estimates two state variables—**position** (x̂) and **velocity** (v̂)—using a predict-then-correct cycle. For each new measurement, it first predicts where the target should be based on the previous estimate, then corrects this prediction using the actual measurement.

**Prediction equations** (state extrapolation):
$$\hat{x}_{k|k-1} = \hat{x}_{k-1|k-1} + \Delta T \cdot \hat{v}_{k-1|k-1}$$
$$\hat{v}_{k|k-1} = \hat{v}_{k-1|k-1}$$

**Update equations** (correction using measurement residual $r_k = z_k - \hat{x}_{k|k-1}$):
$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + \alpha \cdot r_k$$
$$\hat{v}_{k|k} = \hat{v}_{k|k-1} + \frac{\beta}{\Delta T} \cdot r_k$$

The parameter **α controls position correction** (how much the filter trusts new measurements vs. its prediction), while **β controls velocity correction** (how quickly the filter adapts its velocity estimate). Valid ranges for stability require **0 < α < 1** and **0 < β < 4 − 2α**.

The filter relates directly to exponential smoothing: setting β = 0 reduces it to single exponential smoothing, while the full α-β structure parallels Holt's double exponential smoothing. It is also a **special case of the Kalman filter**—specifically, the steady-state Kalman filter with a constant-velocity motion model and fixed noise covariances converges to identical equations.

---

## Mathematical derivation of dynamic error (динамічна похибка)

Dynamic error (also called bias, lag, or truncation error) is the **steady-state difference between the true signal and the filter's estimate** when tracking signals that change faster than the filter's model assumes. Since the α-β filter assumes constant velocity, any acceleration causes tracking lag.

**Physical interpretation**: The filter's recursive averaging inherently delays response. When a target accelerates, the filter's velocity estimate lags behind reality, causing position estimates to fall progressively behind the true position. This gap persists throughout the acceleration period.

**Mathematical derivation using transfer function analysis**: The filter's z-domain transfer function is:
$$G(z) = \frac{\alpha \cdot z}{z^2 + (\alpha + \beta - 2)z + (1 - \alpha)}$$

Applying the Final Value Theorem to a parabolic input (constant acceleration $a_s$) yields the **steady-state dynamic error**:

$$\boxed{e_{dynamic} = \frac{a_s \cdot T^2}{\beta}}$$

where $T$ is the sampling interval. This formula reveals critical dependencies:

- **Inverse relationship with β**: Doubling β halves the steady-state lag. Larger β means faster velocity adaptation, reducing lag.
- **Quadratic dependence on sampling interval**: Doubling the time between measurements quadruples the error. More frequent sampling dramatically reduces lag.
- **Linear dependence on acceleration**: Faster-changing signals produce proportionally larger errors.

For **constant velocity signals**, the α-β filter achieves zero steady-state error (the model matches reality). For **constant position**, error is also zero. The dynamic error only manifests when signal dynamics exceed the model order—hence why the α-β-γ filter (adding acceleration estimation) eliminates steady-state error for constant acceleration.

---

## Mathematical derivation of stochastic error (стохастична похибка)

Stochastic error quantifies how measurement noise propagates through the filter to contaminate estimates. Unlike deterministic dynamic error, stochastic error is random but characterized by its variance.

**Noise propagation mechanism**: Each measurement $z_k$ contains true position plus noise $\nu_k$ with variance $\sigma_v^2$. The update equations inject noise into estimates: the term $\alpha \cdot r_k$ adds noise to position estimates, while $(\beta/\Delta T) \cdot r_k$ adds noise to velocity estimates.

The **noise ratio ρ** measures how much measurement noise passes through to the output:

$$\boxed{\rho^2 = \frac{2\alpha^2 + \alpha\beta + 2\beta}{\alpha(4 - \beta - 2\alpha)}}$$

For the simpler single-state alpha filter (exponential smoothing), the **variance reduction ratio** is:
$$\text{VRR} = \frac{\sigma_{output}^2}{\sigma_{input}^2} = \frac{\alpha}{2-\alpha}$$

This yields the **equivalent number of samples** being averaged:
$$N_{equiv} = \frac{2-\alpha}{\alpha}$$

For α = 0.1, this equals 19 samples—the filter averages roughly the last 19 measurements. For α = 0.5, only about 3 samples are effectively averaged, providing much less smoothing.

| α value | Variance Reduction | Equivalent Samples |
|---------|-------------------|-------------------|
| 0.1 | 5.3% of input | ~19 samples |
| 0.2 | 11.1% of input | ~9 samples |
| 0.5 | 33.3% of input | ~3 samples |
| 0.8 | 66.7% of input | ~1.5 samples |

**Frequency response perspective**: The α-β filter acts as a low-pass filter. Smaller α narrows the bandwidth, attenuating high-frequency noise but also slowing response to legitimate signal changes.

---

## The fundamental trade-off between accuracies

The statement "dynamic and stochastic accuracy are relatively low" reflects a **fundamental constraint**: no choice of α and β can optimize both simultaneously. This is the filtering equivalent of the bias-variance trade-off in statistics.

**Mathematical proof of inverse dependency**:

The dynamic (lag) error for acceleration is:
$$e_{dynamic} = \frac{a_s \cdot T^2}{\beta} \quad \text{(decreases with larger β)}$$

The stochastic (noise) error characterized by the noise ratio:
$$\rho^2 = \frac{2\alpha^2 + \alpha\beta + 2\beta}{\alpha(4 - \beta - 2\alpha)} \quad \text{(increases with larger α, β)}$$

**To minimize dynamic error**: increase β → more noise passes through (worse stochastic accuracy)  
**To minimize stochastic error**: decrease α and β → larger lag for changing signals (worse dynamic accuracy)

The **total mean square error** combines both components:
$$\text{MSE}_{total} = \text{Bias}^2 + \text{Variance} = e_{dynamic}^2 + \rho^2 \sigma_v^2$$

Minimizing this sum requires **balancing** the two terms, not eliminating either. The optimal balance depends on signal-to-noise ratio and expected target dynamics.

**Optimal parameter relationships** have been derived for this trade-off:

- **Benedict-Bordner criterion** (1962): $\beta = \frac{\alpha^2}{2-\alpha}$ minimizes combined transient and steady-state error for constant-velocity targets
- **Kalata's tracking index** (1984): $\lambda = \frac{\sigma_w T^2}{\sigma_v}$ yields optimal $\alpha = \frac{-\lambda^2 + \sqrt{\lambda^4 + 16\lambda^2}}{8}$

The "relatively low" accuracy occurs because any compromise position on this trade-off curve yields moderate performance in both dimensions. The Kalman filter, with time-varying gains adapted to current uncertainty, can navigate this trade-off more optimally—but at steady-state, it converges to the same α-β structure.

---

## Python implementation for experimentation

The following implementation demonstrates the filter and allows you to observe the trade-off directly:

```python
import numpy as np
import matplotlib.pyplot as plt

def alpha_beta_filter(measurements, alpha, beta, dt=1.0):
    """
    Alpha-Beta filter implementation.
    
    Parameters:
        measurements: array of noisy position measurements
        alpha: position correction gain (0 < alpha < 1)
        beta: velocity correction gain (0 < beta < 4-2*alpha)
        dt: time step between measurements
    
    Returns:
        x_est: estimated positions
        v_est: estimated velocities
    """
    n = len(measurements)
    x_est = np.zeros(n)
    v_est = np.zeros(n)
    
    # Initialize with first measurement
    x_est[0] = measurements[0]
    v_est[0] = 0.0
    
    for k in range(1, n):
        # PREDICT: extrapolate using motion model
        x_pred = x_est[k-1] + v_est[k-1] * dt
        v_pred = v_est[k-1]
        
        # RESIDUAL: difference between measurement and prediction
        residual = measurements[k] - x_pred
        
        # UPDATE: correct prediction using measurement
        x_est[k] = x_pred + alpha * residual
        v_est[k] = v_pred + (beta / dt) * residual
    
    return x_est, v_est


def benedict_bordner_beta(alpha):
    """Optimal beta using Benedict-Bordner criterion."""
    return alpha**2 / (2 - alpha)


def noise_ratio(alpha, beta):
    """Calculate noise ratio ρ² for given parameters."""
    return (2*alpha**2 + alpha*beta + 2*beta) / (alpha * (4 - beta - 2*alpha))


def steady_state_lag(acceleration, T, beta):
    """Calculate steady-state dynamic error for constant acceleration."""
    return acceleration * T**2 / beta


# Demonstration: Trade-off visualization
def demonstrate_tradeoff():
    np.random.seed(42)
    n = 300
    dt = 1.0
    
    # True signal: constant velocity, then constant acceleration
    true_pos = np.zeros(n)
    velocity = 2.0
    acceleration = 0.3
    
    for k in range(1, n):
        if k < 100:
            true_pos[k] = true_pos[k-1] + velocity * dt
        else:
            t = k - 100
            true_pos[k] = true_pos[99] + velocity*t*dt + 0.5*acceleration*(t*dt)**2
    
    # Add measurement noise
    noise_std = 8.0
    measurements = true_pos + np.random.normal(0, noise_std, n)
    
    # Test different alpha values
    alphas = [0.1, 0.3, 0.6, 0.9]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, alpha in zip(axes.flat, alphas):
        beta = benedict_bordner_beta(alpha)
        x_est, v_est = alpha_beta_filter(measurements, alpha, beta, dt)
        
        # Calculate metrics
        rho2 = noise_ratio(alpha, beta)
        lag = steady_state_lag(acceleration, dt, beta)
        rmse = np.sqrt(np.mean((x_est[150:] - true_pos[150:])**2))
        
        ax.plot(true_pos, 'g-', linewidth=2, label='True')
        ax.plot(measurements, 'b.', alpha=0.2, markersize=2, label='Measurements')
        ax.plot(x_est, 'r-', linewidth=1.5, label='Filtered')
        ax.axvline(100, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'α={alpha:.1f}, β={beta:.3f}\n'
                    f'Noise ratio ρ²={rho2:.3f}, Lag error={lag:.1f}, RMSE={rmse:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('alpha_beta_tradeoff.png', dpi=150)
    plt.show()

# Run: demonstrate_tradeoff()
```

This code generates a signal with constant velocity followed by constant acceleration, adds noise, and filters with different α values. You'll observe:

- **Low α (0.1)**: Smooth estimates but significant lag during acceleration
- **High α (0.9)**: Tracks changes quickly but estimates are noisy
- **Middle values**: Compromise with moderate performance on both metrics

For tracking accelerating targets, the **alpha-beta-gamma filter** adds a third state (acceleration) and gain γ, eliminating steady-state lag for constant acceleration:

```python
def alpha_beta_gamma_filter(measurements, alpha, beta, gamma, dt=1.0):
    """Alpha-Beta-Gamma filter for tracking with acceleration."""
    n = len(measurements)
    x_est = np.zeros(n)
    v_est = np.zeros(n)
    a_est = np.zeros(n)
    
    x_est[0] = measurements[0]
    
    for k in range(1, n):
        # Predict with acceleration
        x_pred = x_est[k-1] + v_est[k-1]*dt + 0.5*a_est[k-1]*dt**2
        v_pred = v_est[k-1] + a_est[k-1]*dt
        a_pred = a_est[k-1]
        
        residual = measurements[k] - x_pred
        
        # Update all three states
        x_est[k] = x_pred + alpha * residual
        v_est[k] = v_pred + (beta/dt) * residual
        a_est[k] = a_pred + (2*gamma/dt**2) * residual
    
    return x_est, v_est, a_est
```

---

## Conclusion

The alpha-beta filter's elegance lies in its simplicity—two parameters governing a recursive prediction-correction cycle. However, this simplicity imposes the fundamental trade-off your lecture referenced. The **noise ratio ρ² = (2α² + αβ + 2β)/[α(4-β-2α)]** and **steady-state lag e = a·T²/β** are mathematically coupled: improving one necessarily degrades the other.

When both accuracies appear "relatively low," it typically indicates either suboptimal parameter selection, a mismatch between filter order and signal dynamics (using α-β for accelerating targets), or simply the inherent limitation that fixed-gain filters cannot adapt to changing conditions. The Kalman filter addresses this through time-varying gains computed from uncertainty estimates, but at steady-state converges to the same structure. Understanding this trade-off is foundational for all recursive estimation—the same principle extends to exponential smoothing in time series, control system observers, and signal processing filters generally.