import numpy as np
import pandas as pd 

def solve_H_right_inverse(
    Y, H, tol_clip=1e-12, major_tol=1e-2,
    verbose=True, auto_orient=True, renormalize_rows=False
):
    """
    Solve W from Y = W H with affine sum to one via right inverse on [H, 1].
    Returns (W, major_mask, diag).
    Always clips to the simplex and reports major violations before clipping.
    """
    H_in = np.array(H, dtype=float, copy=True)

    # auto orientation so that rows of H sum to one
    transposed = False
    if auto_orient:
        row_dev = np.max(np.abs(H_in.sum(axis=1) - 1.0))
        col_dev = np.max(np.abs(H_in.sum(axis=0) - 1.0))
        if col_dev < row_dev:
            H_in = H_in.T
            transposed = True

    if renormalize_rows:
        rs = H_in.sum(axis=1, keepdims=True)
        rs[rs == 0.0] = 1.0
        H_in = H_in / rs

    K, J = H_in.shape
    n = Y.shape[0]

    # augment to encode sum to one
    H_aug = np.hstack([H_in, np.ones((K, 1))])      # K by J+1
    Y_aug = np.hstack([Y,    np.ones((n, 1))])      # n by J+1

    # right inverse via SVD pseudoinverse
    H_aug_R = np.linalg.pinv(H_aug)                  # (J+1) by K

    # raw weights and diagnostics
    W_raw = Y_aug @ H_aug_R                          # n by K
    sum_dev = np.abs(W_raw.sum(axis=1) - 1.0)
    min_neg = W_raw.min(axis=1)
    major_mask = (min_neg < -major_tol) | (sum_dev > major_tol)
    major_count = int(np.count_nonzero(major_mask))

    # print diagnostics
    if verbose:
        G = H_aug @ H_aug.T
        try:
            condG = float(np.linalg.cond(G))
        except np.linalg.LinAlgError:
            condG = float("inf")
        I_err = float(np.linalg.norm(H_aug @ H_aug_R - np.eye(K), ord=np.inf))
        aug_resid = float(np.linalg.norm(Y_aug - W_raw @ H_aug, ord=np.inf))
#         print(f"{major_count} of {n} rows had major simplex violations before clipping")
#         print(f"H transposed: {transposed}")
#         print(f"max row sum dev of H: {np.max(np.abs(H_in.sum(axis=1)-1.0)):.3e}")
#         print(f"rank of H_aug: {np.linalg.matrix_rank(H_aug)} of {K}")
#         print(f"cond number of G: {condG:.3e}")
        print(f"||H_aug H_aug_R - I||_inf: {I_err:.3e}")
#         print(f"augmented residual inf norm: {aug_resid:.3e}")

    # clip and renormalize to the simplex
    W = np.maximum(W_raw, tol_clip)
    s = W.sum(axis=1, keepdims=True)
    zero_rows = (s[:, 0] == 0.0)
    if np.any(zero_rows):
        W[zero_rows] = 1.0 / K
        s[zero_rows] = 1.0
    W /= s

    diag = {
        "transposed": transposed,
        "max_row_sum_dev_H": float(np.max(np.abs(H_in.sum(axis=1)-1.0))),
        "rank_H_aug": int(np.linalg.matrix_rank(H_aug)),
        "cond_G": condG,
        "I_err": I_err,
        "aug_resid_inf": aug_resid,
        "major_count": major_count,
    }
    return W, major_mask, diag

# ---------- NNLS helpers ----------
def _nnls_pg_batch(A, B, max_iter=400, tol=1e-7):
    """
    Solve min_X ||A X - B||_F^2 s.t. X >= 0, PGD (batch).
    A: (m x k), B: (m x n)  -> X: (k x n)
    """
    AT = A.T
    ATA = AT @ A
    ATB = AT @ B
    L = np.linalg.norm(ATA, 2) + 1e-12
    step = 1.0 / L
    X = np.zeros((A.shape[1], B.shape[1]))
    for _ in range(max_iter):
        grad = ATA @ X - ATB
        X_new = np.maximum(X - step * grad, 0.0)
        if np.linalg.norm(X_new - X, 'fro') <= tol * (1 + np.linalg.norm(X, 'fro')):
            X = X_new
            break
        X = X_new
    return X

def _nnls_batch(A, B, prefer_scipy=True):
    """
    Try SciPy's nnls column-by-column; fall back to PGD if SciPy unavailable.
    A: (m x k), B: (m x n) -> X: (k x n)
    """
    if prefer_scipy:
        try:
            from scipy.optimize import nnls
            X = np.zeros((A.shape[1], B.shape[1]))
            for j in range(B.shape[1]):
                x, _ = nnls(A, B[:, j])
                X[:, j] = x
            return X
        except Exception:
            pass
    return _nnls_pg_batch(A, B)

def xray_BJ(Y, K, seed=123, normalize=False, prefer_scipy_nnls=True):
    """
    XRAY on Y (N x J). 
    Returns:
      H  : (K x J) endmembers (rows are selected samples' spectra)
      idx: (K,) indices of selected rows in Y
    normalize: True | False  (column normalization before selection)
    """
    
    rng = np.random.default_rng(seed)
    
    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()
        
    # columns = samples for the cone ops
    V = Y.T.copy()                # (J x N)
    if normalize:
        V /= (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    
    N = V.shape[1]
    idx = []

    # 1) first anchor: max column norm
    j0 = int(np.argmax(np.sum(V * V, axis=0)))
    idx.append(j0)

    # 2) grow the cone to K anchors
    for t in range(1, K):
        U = V[:, idx]                              # (J x t)
        # project all points onto cone(U): solve min_H ||U H - V||_F^2, H>=0
        Hcoef = _nnls_batch(U, V, prefer_scipy_nnls)     # (t x N)
        R = V - U @ Hcoef                           # residuals (J x N)
        rnorm = np.linalg.norm(R, axis=0)
        rnorm[idx] = -np.inf                        # avoid reselecting anchors
        j_new = int(np.argmax(rnorm))
        idx.append(j_new)

    H = V[:, idx].T                                 # (K x J)
    # Map indices back to Y rows (since V columns = Y rows)
    return H, np.array(idx, dtype=int)