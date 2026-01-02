# Conformal Density Parameter: Computational Supplement

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the computational supplement for:

> **"On the Necessity of Internal Degrees of Freedom in Conformally Invariant Structures"**  
> Kerym Makraini  
> *Journal of Mathematical Physics* (submitted)

The code provides numerical verification of the main theoretical results concerning the **conformal density parameter** ρ, including:

- **Theorem 5.4**: Existence and convergence of the regularized limit
- **Theorem 5.5**: Conformal invariance under PSL(2,ℂ)
- **Lemma 5.3**: Cutoff trace convergence
- **Appendix D**: Explicit variable ρ configurations

## Repository Structure

```
conformal-density-repo/
├── src/
│   └── conformal_density.py    # Main computational module
├── tests/
│   └── test_conformal_density.py   # Comprehensive test suite
├── notebooks/
│   └── generate_figures.ipynb  # Figure generation
├── figures/
│   └── (generated figures)
├── data/
│   └── (output data)
├── requirements.txt
└── README.md
```

## Installation

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Matplotlib ≥ 3.4
- pytest ≥ 6.0 (for testing)

### Setup

```bash
# Clone repository
git clone https://github.com/KerymMacryn/conformal-density-repo.git
cd conformal-density

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
pytest tests/ -v
```

## Quick Start

### Basic Usage

```python
from src.conformal_density import ConformalDensity, rho_radial
import numpy as np

# Create conformal density calculator for helicity n=0
cd = ConformalDensity(helicity=0, cutoff_type='gaussian')

# Compute ρ for given cutoff and hemisphere fraction
Lambda = 50.0  # Cutoff parameter
rho = cd.compute_trace_ratio(Lambda, hemisphere_fraction=0.8)
print(f"Conformal density ρ = {rho:.6f}")

# Verify convergence
Lambda_values = np.array([10, 20, 50, 100, 200])
ratios = cd.verify_convergence(Lambda_values, hemisphere_fraction=0.8)
print(f"Convergence: {ratios}")

# Evaluate explicit variable ρ example
x = np.array([[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]])  # Spacetime points
rho_values = rho_radial(x, R=1.0, n=0)
print(f"Radial ρ values: {rho_values}")
```

### Running Tests

```bash
# Run all tests
pytest tests/test_conformal_density.py -v

# Run specific test class
pytest tests/test_conformal_density.py::TestTheoremExistence -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Generating Figures

```bash
cd notebooks
jupyter notebook generate_figures.ipynb
```

Or run programmatically:

```python
from notebooks.generate_figures import generate_all_figures
generate_all_figures()
```

## Main Components

### 1. Fourier-Mellin Transform (`FourierMellinTransform`)

Implements the Fourier-Mellin transform on CP¹ for cohomology regularization:

```
f̂(s, m) = ∫∫ f(r·e^{iθ}) r^{-s-1} e^{-imθ} r dr dθ
```

### 2. Spectral Cutoff (`SpectralCutoff`)

Implements the cutoff operator χ_Λ = φ(M/Λ) with options:
- `'gaussian'`: φ(x) = exp(-x²)
- `'sharp'`: φ(x) = χ_{|x|≤1}
- `'smooth'`: Smooth bump function

### 3. Cohomology Space (`CohomologySpace`)

Represents H¹(CP¹, O(-n-2)) with:
- Dimension formula: dim = n + 1
- Basis function evaluation
- Projector matrix computation

### 4. Conformal Density (`ConformalDensity`)

Main class computing:
```
ρ(x) := lim_{ε→0⁺} Tr(χ_{Λ(ε)} P_{H_x,ε}) / Tr(χ_{Λ(ε)} P_x)
```

### 5. PSL(2,ℂ) Transformations (`PSL2C_Action`)

Implements Möbius transformations for conformal invariance tests:
- Rotations
- Boosts
- Dilations
- General SL(2,ℂ) matrices

## Numerical Results

### Theorem Verification Summary

| Theorem | Test | Status |
|---------|------|--------|
| 5.4 (Existence) | Limit exists for n=0,1,2,3 | ✅ Verified |
| 5.4 (Independence) | Independent of cutoff type | ✅ Verified |
| 5.4 (Vacuum) | ρ = 1 in Minkowski vacuum | ✅ Verified |
| 5.5 (Invariance) | PSL(2,ℂ) invariance | ✅ Verified |
| Lemma 5.3 | Trace convergence | ✅ Verified |
| Lemma 5.4 | Projector stability | ✅ Verified |

### Convergence Results

For helicity n=0 (scalar field):

| Λ | Tr(χ_Λ P) | Rel. Error |
|---|-----------|------------|
| 10 | 0.85 | 0.15 |
| 50 | 0.96 | 0.04 |
| 100 | 0.98 | 0.02 |
| 200 | 0.99 | 0.01 |


## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Contact

- **Author**: Kerym Makraini
- **Email**: mhamed34@alumno.uned.es
- **Institution**: UNED, Madrid, Spain

## Acknowledgments

This work was supported by [funding information]. The author thanks [acknowledgments] for valuable discussions.

---

**Last updated**: January 2025
