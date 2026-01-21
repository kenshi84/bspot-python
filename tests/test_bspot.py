import bspot
import numpy as np
import pytest

def test_compute_matching():
    A = np.random.randn(3,10000)
    B = np.random.randn(3,10000)
    bspot.set_num_threads(1)
    bspot.compute_matching(A, B)
    bspot.set_num_threads(0)
    bspot.compute_matching(A, B, orthogonal=True)
    bspot.compute_matching(A, B, gaussian=True)
    with pytest.raises(RuntimeError):
        bspot.compute_matching(A, B, orthogonal=True, gaussian=True)
    with pytest.raises(RuntimeError):
        B_invalid = np.random.randn(2,10000)
        bspot.compute_matching(A, B_invalid)
    with pytest.raises(RuntimeError):
        B_invalid = np.random.randn(3,20000)
        bspot.compute_matching(A, B_invalid)

def test_compute_coupling():
    A = np.random.randn(3,10000)
    B = np.random.randn(3,20000)
    mu = np.random.randn(A.shape[1]) ** 2
    nu = np.random.randn(B.shape[1]) ** 2
    mu /= mu.sum()
    nu /= nu.sum()
    bspot.compute_coupling(A, mu, B, nu)
    with pytest.raises(RuntimeError):
        B_invalid = np.random.randn(2,20000)
        bspot.compute_coupling(A, mu, B_invalid, nu)
    with pytest.raises(RuntimeError):
        mu_invalid = np.random.randn(A.shape[1] + 1) ** 2
        mu_invalid /= mu_invalid.sum()
        bspot.compute_coupling(A, mu_invalid, B, nu)
    with pytest.raises(RuntimeError):
        nu_invalid = np.random.randn(B.shape[1] - 1) ** 2
        nu_invalid /= nu_invalid.sum()
        bspot.compute_coupling(A, mu, B, nu_invalid)

def test_compute_transport_gradient():
    A = np.random.randn(3,10000)
    B = np.random.randn(3,20000)
    mu = np.random.randn(A.shape[1]) ** 2
    nu = np.random.randn(B.shape[1]) ** 2
    mu /= mu.sum()
    nu /= nu.sum()
    bspot.compute_transport_gradient(A, mu, B, nu)
    with pytest.raises(RuntimeError):
        B_invalid = np.random.randn(2,20000)
        bspot.compute_transport_gradient(A, mu, B_invalid, nu)
    with pytest.raises(RuntimeError):
        mu_invalid = np.random.randn(A.shape[1] + 1) ** 2
        mu_invalid /= mu_invalid.sum()
        bspot.compute_transport_gradient(A, mu_invalid, B, nu)
    with pytest.raises(RuntimeError):
        nu_invalid = np.random.randn(B.shape[1] - 1) ** 2
        nu_invalid /= nu_invalid.sum()
        bspot.compute_transport_gradient(A, mu, B, nu_invalid)

def test_compute_partial_matching():
    A = np.random.randn(3,10000)
    B = np.random.randn(3,20000)
    bspot.compute_partial_matching(A, B)
    with pytest.raises(RuntimeError):
        B_invalid = np.random.randn(2,20000)
        bspot.compute_partial_matching(A, B_invalid)
    with pytest.raises(RuntimeError):
        B_invalid = np.random.randn(3,1000)
        bspot.compute_partial_matching(A, B_invalid)
