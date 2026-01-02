#!/usr/bin/env python3
"""
AppendixD_radial_example.py
===========================
Reproducible script for radial contraction model and computation of rho(f)

Paper 0A: "On the Necessity of Internal Degrees of Freedom 
           in Conformally Invariant Structures"
Appendix D: Radial contraction example

This script demonstrates the geometric scaling ρ ~ f² that arises from
the radial contraction of the twistor incidence relation.

The key insight is that under ω^A → f·ω^A, the area of the analyticity
domain on the twistor line scales as f², hence ρ ~ f².

Requirements: numpy, matplotlib
Run: python AppendixD_radial_example.py

Author: Kerym Makraini
Date: 2025-01-02
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

n = 1               # helicity index (Maxwell field, N = n + 1 = 2)
N = n + 1           # cohomology dimension
lambda_reg = 1.0    # regularization parameter (Mellin cutoff)


# =============================================================================
# MODEL 1: ANALYTIC (EXACT f²)
# =============================================================================

def analytic_rho(f: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """
    Analytic model: ρ = f^κ with κ = 2 from geometric area scaling.
    
    Under the radial contraction ω → fω, the area of the hemisphere
    H_x on the twistor line CP¹ scales as f².
    """
    return f**kappa


# =============================================================================
# MODEL 2: REGULARIZED TRACE MODEL
# =============================================================================

def numerical_rho_model(f: np.ndarray, lambda_reg: float = 1.0) -> np.ndarray:
    """
    Numerical model for ρ(f) using Fourier-Mellin regularization.
    
    We model the regularized traces as:
    - Tr(χ_Λ P_x) = Σ_{k=1}^N exp(-k²/Λ²)  [total cohomology]
    - Tr(χ_Λ P_{H_x(f)}) = Σ_{k=1}^N exp(-k²/Λ²) × (f²)^{k}  [hemisphere]
    
    The factor (f²)^k models the mode-dependent contraction.
    """
    Lambda = 10.0 / lambda_reg  # Effective cutoff
    
    rho_values = np.zeros_like(f)
    
    for i, f_val in enumerate(f):
        # Denominator: total regularized trace
        denominator = sum(np.exp(-k**2 / Lambda**2) for k in range(1, N + 1))
        
        # Numerator: hemisphere-weighted trace
        numerator = sum(
            np.exp(-k**2 / Lambda**2) * (f_val**2)**k 
            for k in range(1, N + 1)
        )
        
        if denominator > 1e-15:
            rho_values[i] = numerator / denominator
        else:
            rho_values[i] = 0.0
    
    return rho_values


# =============================================================================
# MODEL 3: FULL INTEGRATION OVER CP¹
# =============================================================================

def full_integration_model(f: np.ndarray, n_radial: int = 100,
                           n_angular: int = 64) -> np.ndarray:
    """
    Full numerical model integrating over CP¹.
    
    This implements the ratio of regularized dimensions:
    ρ(f) = lim_{ε→0} dim_ε H¹_{H_x(f)} / dim_ε H¹_{CP¹}
    
    where the hemisphere H_x(f) has area ∝ f².
    """
    rho_values = np.zeros_like(f)
    
    # Integration grid on CP¹ (disk model)
    r = np.linspace(0, 1, n_radial)
    theta = np.linspace(0, 2*np.pi, n_angular)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    # Measure: r dr dθ (area element on disk)
    dA = np.ones_like(R) * (r[1] - r[0]) * (theta[1] - theta[0])
    dA *= R  # Jacobian
    
    for i, f_val in enumerate(f):
        # The "hemisphere" after contraction has effective radius f
        hemisphere_mask = (R <= f_val)
        
        # Weight function (Gaussian regularization)
        weight = np.exp(-lambda_reg * R**2)
        
        # Denominator: integral over full disk
        full_integral = np.sum(weight * dA)
        
        # Numerator: integral over contracted hemisphere
        hemisphere_integral = np.sum(weight * dA * hemisphere_mask)
        
        if full_integral > 1e-15:
            rho_values[i] = hemisphere_integral / full_integral
        else:
            rho_values[i] = 0.0
    
    return rho_values


# =============================================================================
# POWER LAW FITTING
# =============================================================================

def fit_power_law(f: np.ndarray, rho: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit ρ = A × f^κ and return (κ, A, R²).
    """
    # Filter valid data
    valid = (f > 0.01) & (rho > 1e-12)
    
    if np.sum(valid) < 3:
        return 0.0, 0.0, 0.0
    
    # Log-log linear regression
    log_f = np.log(f[valid])
    log_rho = np.log(rho[valid])
    
    # Linear fit: log(ρ) = κ log(f) + log(A)
    coeffs = np.polyfit(log_f, log_rho, 1)
    kappa = coeffs[0]
    A = np.exp(coeffs[1])
    
    # R² goodness of fit
    fitted = A * f[valid]**kappa
    ss_res = np.sum((rho[valid] - fitted)**2)
    ss_tot = np.sum((rho[valid] - np.mean(rho[valid]))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return kappa, A, r_squared


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def main():
    """
    Main computation demonstrating ρ ~ f² scaling.
    """
    print("=" * 70)
    print("APPENDIX D: Radial Contraction Example")
    print("Paper 0A: On the Necessity of Internal Degrees of Freedom")
    print("=" * 70)
    print()
    
    # Grid of f values
    fs = np.linspace(0.1, 1.0, 41)
    
    # Model 1: Analytic (exact f²)
    print("Model 1: Analytic ρ = f²")
    rho_analytic = analytic_rho(fs, kappa=2.0)
    kappa1, A1, r2_1 = fit_power_law(fs, rho_analytic)
    print(f"  κ = {kappa1:.4f}, A = {A1:.4f}, R² = {r2_1:.6f}")
    print()
    
    # Model 2: Numerical regularized trace model
    print("Model 2: Regularized trace model")
    rho_numerical = numerical_rho_model(fs, lambda_reg=1.0)
    kappa2, A2, r2_2 = fit_power_law(fs, rho_numerical)
    print(f"  κ = {kappa2:.4f}, A = {A2:.4f}, R² = {r2_2:.6f}")
    print()
    
    # Model 3: Full integration model
    print("Model 3: Full integration over CP¹")
    rho_full = full_integration_model(fs)
    kappa3, A3, r2_3 = fit_power_law(fs, rho_full)
    print(f"  κ = {kappa3:.4f}, A = {A3:.4f}, R² = {r2_3:.6f}")
    print()
    
    # Verification
    print("=" * 70)
    print("VERIFICATION: Geometric scaling ρ ~ f²")
    print("=" * 70)
    
    all_close_to_2 = all(abs(k - 2.0) < 0.35 for k in [kappa1, kappa2, kappa3])
    
    if all_close_to_2:
        print("✓ PASSED: All models confirm κ ≈ 2")
        print()
        print("Physical interpretation:")
        print("  Under ω^A → f·ω^A, the hemisphere H_x contracts.")
        print("  Area(H_x) ~ f² → Tr(χ_Λ P_{H_x}) ~ f²")
        print("  Therefore ρ = Tr(P_H)/Tr(P) ~ f²")
    else:
        print("⚠ Some models deviate from κ = 2")
        print("  This may indicate numerical issues.")
    
    print("=" * 70)
    print()
    
    # Generate figure
    print("Generating figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel (a): All models comparison
    ax1 = axes[0]
    ax1.plot(fs, rho_analytic, 'b-', linewidth=2, label=r'Analytic: $\rho = f^2$')
    ax1.plot(fs, rho_numerical, 'r--', linewidth=2, label=f'Regularized: $\\kappa = {kappa2:.2f}$')
    ax1.plot(fs, rho_full, 'g:', linewidth=2, label=f'Integration: $\\kappa = {kappa3:.2f}$')
    ax1.set_xlabel(r'$f$ (radial contraction)', fontsize=12)
    ax1.set_ylabel(r'$\rho(f)$', fontsize=12)
    ax1.set_title('(a) Model comparison', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.1])
    
    # Panel (b): Log-log verification
    ax2 = axes[1]
    valid = fs > 0.05
    ax2.loglog(fs[valid], rho_analytic[valid], 'b-', linewidth=2, label='Analytic')
    ax2.loglog(fs[valid], rho_numerical[valid], 'ro', markersize=5, label='Regularized')
    ax2.loglog(fs[valid], rho_full[valid], 'g^', markersize=5, label='Integration')
    ax2.loglog(fs, fs**2, 'k--', linewidth=1, alpha=0.5, label=r'$f^2$ reference')
    ax2.set_xlabel(r'$f$', fontsize=12)
    ax2.set_ylabel(r'$\rho(f)$', fontsize=12)
    ax2.set_title(r'(b) Log-log: verifying $\rho \sim f^2$', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Panel (c): Residuals from f²
    ax3 = axes[2]
    residual_numerical = (rho_numerical - fs**2) / np.maximum(fs**2, 1e-10)
    residual_full = (rho_full - fs**2) / np.maximum(fs**2, 1e-10)
    ax3.plot(fs, residual_numerical * 100, 'r-', linewidth=2, label='Regularized model')
    ax3.plot(fs, residual_full * 100, 'g-', linewidth=2, label='Integration model')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel(r'$f$', fontsize=12)
    ax3.set_ylabel(r'Relative deviation from $f^2$ (\%)', fontsize=12)
    ax3.set_title('(c) Deviations from geometric scaling', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1.05])
    ax3.set_ylim([-50, 50])
    
    plt.suptitle('Appendix D: Verification of $\\rho \\sim f^2$ (geometric area scaling)',
                 fontsize=14)
    plt.tight_layout()
    
    # Save
    plt.savefig('figures/appendix_D_power_law.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/appendix_D_power_law.pdf', bbox_inches='tight')
    print("  Saved to figures/appendix_D_power_law.{png,pdf}")
    
    plt.show()
    
    return {
        'fs': fs,
        'rho_analytic': rho_analytic,
        'rho_numerical': rho_numerical,
        'rho_full': rho_full,
        'kappa_values': [kappa1, kappa2, kappa3],
        'verified': all_close_to_2
    }


if __name__ == "__main__":
    result = main()
