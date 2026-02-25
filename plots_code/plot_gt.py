import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def snr(k,sigma,t):
    return k(t)/(k(t)**2 + sigma**2 )**0.5

def ve_snr(t):
    std = 0.01 * (50.0 / 0.001) ** t
    return (1 + std**2) ** (-0.5)

def vp_snr(t):
    beta_0 = 0.1
    beta_1 = 20.0
    log_mean_coeff = -0.25 * t**2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    k = np.exp(2 * log_mean_coeff)
    std = 1 - k
    return k / (k**2 + std**2) ** 0.5


t = np.linspace(0, 1, 1000)
# Compute schedules
y_ve = ve_snr(t)
y_vp = vp_snr(t)

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
fig.subplots_adjust(wspace=0.4)  # increase horizontal spacing

border = 0.05

axes[0].plot(t, y_vp, color='navy', lw=1.5, alpha=0.98)  # Swapped order
axes[0].set_title('VP: g(t) (SNR)')  # Swapped title
axes[0].set_xlabel('t')
axes[0].set_ylabel('g(t)')
axes[0].set_xlim(-border, 1+border)
axes[0].set_ylim(-border, 1+border)
axes[0].grid(True, ls='--', alpha=0.5)
axes[0].set_aspect('equal', adjustable='box')

axes[1].plot(t, y_ve, color='navy', lw=1.5, alpha=0.98)  # Swapped order
axes[1].set_title('VE: g(t) (SNR)')  # Swapped title
axes[1].set_xlabel('t')
axes[1].set_ylabel('g(t)')
axes[1].set_xlim(-border, 1+border)
axes[1].set_ylim(-border, 1+border)
axes[1].grid(True, ls='--', alpha=0.5)
axes[1].set_aspect('equal', adjustable='box')

# fig.tight_layout()  # removed to respect wspace

# Save to the path used in LaTeX
out_path = Path(__file__).resolve().parents[1] / "articles/presentazione_aaai25/images/g(t).pdf"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, bbox_inches='tight')

plt.show()


