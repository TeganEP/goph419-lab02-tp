# tests.py
import numpy as np
from src.linalg_interp import gauss_iter_solve, cubic_spline


def test_gauss_iter_solve():
    A = np.array([[4, 1], [1, 3]])
    b = np.array([1, 2])
    x_expected = np.linalg.solve(A, b)
    x_seidel = gauss_iter_solve(A, b, tol=1e-8, alg='seidel')
    np.testing.assert_allclose(x_seidel, x_expected, rtol=1e-5)
    print("Gauss-Seidel test passed!")


def test_spline_function():
    xd = np.array([0, 1, 2, 3])
    yd = xd ** 2
    spline = cubic_spline(xd, yd)
    x_test = np.array([0.5, 1.5, 2.5])
    y_expected = x_test ** 2
    y_test = spline(x_test)
    np.testing.assert_allclose(y_test, y_expected, rtol=1e-5)
    print("Cubic spline test passed!")


if __name__ == "__main__":
    test_gauss_iter_solve()
    test_spline_function()
