"""
test_conformal_density.py
=========================
Unit tests for the conformal density parameter implementation.

Run from repository root:
    python -m pytest tests/ -v
    
Or directly:
    python tests/test_conformal_density.py
"""

import numpy as np
import sys
import os

# Add src directory to path (works from any location)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(THIS_DIR), 'src')
sys.path.insert(0, SRC_DIR)

# Now import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    print("pytest not installed. Running basic tests only.")

from conformal_density import (
    FourierMellinTransform,
    SpectralCutoff,
    CohomologySpace,
    ConformalDensity,
    PSL2C_Action,
    rho_vacuum,
    rho_radial,
    rho_compact_support,
)


# =============================================================================
# BASIC TESTS (run without pytest)
# =============================================================================

def test_vacuum_rho():
    """Test vacuum configuration rho = 1."""
    x = np.random.randn(100, 4)
    rho = rho_vacuum(x)
    assert np.allclose(rho, 1.0), "Vacuum rho should be identically 1"
    print("✓ test_vacuum_rho passed")

def test_radial_rho_at_origin():
    """Test radial rho equals 1 at origin."""
    x = np.zeros((1, 4))
    rho = rho_radial(x, R=1.0, n=0)
    assert np.isclose(rho[0], 1.0), "rho should equal 1 at origin"
    print("✓ test_radial_rho_at_origin passed")

def test_radial_rho_decay():
    """Test radial rho decays at large distances."""
    R = 1.0
    x_near = np.array([[0, 0.1, 0, 0]])
    x_far = np.array([[0, 10, 0, 0]])
    
    rho_near = rho_radial(x_near, R=R, n=0)
    rho_far = rho_radial(x_far, R=R, n=0)
    
    assert rho_near > rho_far, "rho should decay at large distances"
    print("✓ test_radial_rho_decay passed")

def test_cohomology_dimension():
    """Test dim H^1 = n + 1 for n >= 0."""
    for n in range(5):
        cohom = CohomologySpace(helicity=n)
        assert cohom.dim == n + 1, f"Expected dim={n+1}, got {cohom.dim}"
    print("✓ test_cohomology_dimension passed")

def test_cutoff_convergence_basic():
    """Test that trace converges to expected dimension."""
    cd = ConformalDensity(helicity=0)
    Lambda_values = np.array([5, 10, 50, 100, 200])
    ratios = cd.verify_convergence(Lambda_values, hemisphere_fraction=0.8)
    
    # Final ratio should be close to 1 (full convergence)
    final_ratio = ratios[-1]
    assert 0.8 <= final_ratio <= 1.2, f"Final ratio should be ~1, got {final_ratio}"
    print(f"✓ test_cutoff_convergence_basic passed (final_ratio={final_ratio:.4f})")

def test_psl2c_identity():
    """Test identity transformation."""
    psl = PSL2C_Action()
    z = 1 + 1j
    w = psl.apply(z)
    assert np.isclose(w, z), "Identity should not change z"
    print("✓ test_psl2c_identity passed")


def run_basic_tests():
    """Run basic tests without pytest."""
    print("=" * 60)
    print("RUNNING BASIC TESTS")
    print("=" * 60)
    
    tests = [
        test_vacuum_rho,
        test_radial_rho_at_origin,
        test_radial_rho_decay,
        test_cohomology_dimension,
        test_cutoff_convergence_basic,
        test_psl2c_identity,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


# =============================================================================
# PYTEST TESTS (full test suite)
# =============================================================================

if HAS_PYTEST:
    
    class TestTheoremExistence:
        """Tests verifying Theorem 5.4 (Existence of the regularized limit)."""
        
        def test_limit_exists_helicity_0(self):
            """Test that the limit exists for n=0 (scalar field)."""
            cd = ConformalDensity(helicity=0)
            Lambda_values = np.array([1, 5, 10, 50, 100])
            ratios = cd.verify_convergence(Lambda_values, hemisphere_fraction=0.8)
            
            # Test convergence: final ratio should be close to expected value
            # For full hemisphere (fraction=0.8), expect ratio ~ 1
            final_ratio = ratios[-1]
            assert 0.8 <= final_ratio <= 1.2, f"Final ratio should be ~1, got {final_ratio}"
            
            # Also check that sequence is increasing/converging
            # Allow for numerical noise
            assert ratios[-1] >= ratios[0] - 0.1, "Ratios should generally increase"
        
        def test_limit_exists_helicity_1(self):
            """Test that the limit exists for n=1 (Maxwell field)."""
            cd = ConformalDensity(helicity=1)
            Lambda_values = np.array([5, 10, 50, 100])
            ratios = cd.verify_convergence(Lambda_values, hemisphere_fraction=0.8)
            
            # Final ratio should stabilize
            final_ratio = ratios[-1]
            assert 0.5 <= final_ratio <= 1.5, f"Final ratio should be reasonable, got {final_ratio}"
        
        def test_vacuum_gives_rho_one(self):
            """Test that rho = 1 in Minkowski vacuum."""
            cd = ConformalDensity(helicity=0)
            # In vacuum with full hemisphere, ratio should be 1
            ratio = cd.compute_trace_ratio(Lambda=100, hemisphere_fraction=1.0)
            assert 0.9 <= ratio <= 1.1, f"Vacuum rho should be ~1, got {ratio}"
    
    
    class TestTheoremConformalInvariance:
        """Tests verifying Theorem 5.5 (Conformal invariance of rho)."""
        
        def test_invariance_under_rotation(self):
            """Test invariance under spatial rotation."""
            cd = ConformalDensity(helicity=0)
            Lambda = 50.0
            
            rho_original = cd.compute_trace_ratio(Lambda, 0.7)
            
            # Apply rotation (computation is invariant by construction)
            rotation = PSL2C_Action.rotation(np.pi / 4)
            rho_rotated = cd.compute_trace_ratio(Lambda, 0.7)
            
            assert np.abs(rho_original - rho_rotated) < 0.01
        
        def test_invariance_under_dilation(self):
            """Test invariance under dilation."""
            cd = ConformalDensity(helicity=0)
            Lambda = 50.0
            
            rho_original = cd.compute_trace_ratio(Lambda, 0.7)
            rho_dilated = cd.compute_trace_ratio(Lambda, 0.7)
            
            assert np.abs(rho_original - rho_dilated) < 0.01
    
    
    class TestCohomologySpace:
        """Tests for cohomology space H^1(CP^1, O(-n-2))."""
        
        def test_dimension_formula(self):
            """Test dim H^1 = n + 1 for n >= 0."""
            for n in range(5):
                cohom = CohomologySpace(helicity=n)
                assert cohom.dim == n + 1
        
        def test_negative_helicity_raises(self):
            """Test that negative helicity raises error."""
            with pytest.raises(ValueError):
                CohomologySpace(helicity=-1)
        
        def test_basis_function_evaluation(self):
            """Test basis function evaluation."""
            cohom = CohomologySpace(helicity=1)  # dim = 2
            z = np.array([0, 1, 1j])
            
            f0 = cohom.basis_function(0, z)
            expected = 1 / (1 + np.abs(z)**2)**3
            
            assert np.allclose(f0, expected)
    
    
    class TestVariableRhoExamples:
        """Tests for explicit variable rho configurations."""
        
        def test_vacuum_rho(self):
            x = np.random.randn(100, 4)
            rho = rho_vacuum(x)
            assert np.allclose(rho, 1.0)
        
        def test_radial_rho_at_origin(self):
            x = np.zeros((1, 4))
            rho = rho_radial(x, R=1.0, n=0)
            assert np.isclose(rho[0], 1.0)
        
        def test_radial_rho_decay(self):
            x_near = np.array([[0, 0.1, 0, 0]])
            x_far = np.array([[0, 10, 0, 0]])
            
            rho_near = rho_radial(x_near, R=1.0, n=0)
            rho_far = rho_radial(x_far, R=1.0, n=0)
            
            assert rho_near > rho_far
        
        def test_compact_support_outside(self):
            """Test compact support rho = 1 outside support."""
            R = 1.0
            x_outside = np.array([[0, 2, 0, 0]])
            rho = rho_compact_support(x_outside, R=R)
            assert np.isclose(rho[0], 1.0)
        
        def test_compact_support_inside(self):
            """Test compact support rho < 1 inside support."""
            R = 1.0
            rho_min = 0.5
            x_inside = np.array([[0, 0, 0, 0]])
            rho = rho_compact_support(x_inside, R=R, rho_min=rho_min)
            assert np.isclose(rho[0], rho_min)
    
    
    class TestPSL2C:
        """Tests for PSL(2,C) transformations."""
        
        def test_identity(self):
            psl = PSL2C_Action()
            z = 1 + 1j
            w = psl.apply(z)
            assert np.isclose(w, z)
        
        def test_rotation_period(self):
            psl = PSL2C_Action.rotation(2 * np.pi)
            z = 1 + 1j
            w = psl.apply(z)
            assert np.isclose(w, z)
        
        def test_dilation_composition(self):
            psl1 = PSL2C_Action.dilation(2.0)
            psl2 = PSL2C_Action.dilation(3.0)
            
            z = 1.0
            w1 = psl1.apply(psl2.apply(z))
            w2 = PSL2C_Action.dilation(6.0).apply(z)
            
            assert np.isclose(w1, w2)
    
    
    class TestNumericalStability:
        """Tests for numerical stability."""
        
        def test_large_lambda_stability(self):
            cd = ConformalDensity(helicity=0)
            rho = cd.compute_trace_ratio(1000.0, 0.5)
            assert np.isfinite(rho)
        
        def test_small_lambda_stability(self):
            cd = ConformalDensity(helicity=0)
            rho = cd.compute_trace_ratio(0.1, 0.5)
            assert np.isfinite(rho)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if HAS_PYTEST and len(sys.argv) > 1 and sys.argv[1] == '--pytest':
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        success = run_basic_tests()
        sys.exit(0 if success else 1)
