import numpy as np


def gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='seidel'):
    """
    Solves Ax = b using iterative Gauss-Seidel or Jacobi methods.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = A.shape
    if n != m or b.shape[0] != n:
        raise ValueError("Matrix dimensions should match Ax = b.")

    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.asarray(x0, dtype=float)
        if x.shape != b.shape:
            raise ValueError("x0 must match the shape of b.")

    alg = alg.strip().lower()
    if alg not in {'seidel', 'jacobi'}:
        raise ValueError("Algorithm must be 'seidel' or 'jacobi'.")

    max_iter = 10000
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) / np.linalg.norm(x_new, ord=np.inf) < tol:
            return x_new
        x = x_new

    raise RuntimeWarning("Solution did not converge within maximum iterations.")


def cubic_spline(xd, yd, order=3):
    """
    Generates a cubic spline interpolation function.
    """
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    if len(xd) != len(yd):
        raise ValueError("xd and yd must have the same length.")
    if np.any(np.diff(xd) <= 0):
        raise ValueError("xd must be strictly increasing.")

    n = len(xd) - 1
    h = np.diff(xd)

    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    A[0, 0] = A[n, n] = 1
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((yd[i + 1] - yd[i]) / h[i] - (yd[i] - yd[i - 1]) / h[i - 1])

    M = np.linalg.solve(A, b)

    def spline_function(x):
        x = np.asarray(x)
        if np.any(x < xd[0]) or np.any(x > xd[-1]):
            raise ValueError(f"x values must be within the range [{xd[0]}, {xd[-1]}].")

        result = np.zeros_like(x, dtype=float)
        for i in range(n):
            mask = (xd[i] <= x) & (x <= xd[i + 1])
            dx = x[mask] - xd[i]
            a, b, c, d = yd[i], M[i], M[i + 1], (M[i + 1] - M[i]) / (3 * h[i])
            result[mask] = a + b * dx + c * dx ** 2 + d * dx ** 3
        return result

    return spline_function
