import time
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.special import gammaln  # log-factorial
from scipy.special import comb

# Faster 
def log_intrinsic_volume_score(subset, tol=1e-12):
    """
    Fast log-volume for the (K-1)-simplex spanned by 'subset' (K x d).
    Volume = (prod_{i=1..r} s_i) / r!, where s_i are singular values of the edge matrix.
    Returns (-inf, r) if degenerate.
    """
    base = subset[0]
    A = subset[1:] - base           # shape: (K-1, d)
    # SVD of edge matrix
    _, S, _ = np.linalg.svd(A, full_matrices=False)
    r = int((S > tol).sum())
    # For a K-point simplex we expect r == K-1; otherwise degenerate
    if r < A.shape[0]:
        return -np.inf, r
    # log volume = sum(log S) - log(r!)
    log_vol = float(np.log(S[:r]).sum() - gammaln(r + 1))
    return log_vol, r

# def log_intrinsic_volume_score(subset, tol=1e-12):
#     """
#     Compute the log of the intrinsic r-dimensional volume of the convex hull
#     of the given convex independent points, where r = affine rank of (subset - base).

#     Returns:
#         log_volume: float (log of intrinsic volume, or -inf if degenerate)
#         r: int (intrinsic dimension)
#     """
#     base = subset[0]
#     centered = subset - base

#     # Determine intrinsic dimension
#     U, S, Vt = np.linalg.svd(centered, full_matrices=False)
#     r = (S > tol).sum()

#     if r < 1:
#         return -np.inf, r  # all points are identical or colinear

#     # Project to intrinsic r-dimensional subspace
#     basis = Vt[:r]  # shape: (r, d)
#     projected = centered @ basis.T  # shape: (K, r)

#     try:
#         # pass an explicit empty string so no None ever reaches qhull
#         hull = ConvexHull(np.asarray(projected, float), qhull_options="")
#         vol = hull.volume
#         if vol <= 0:
#             return -np.inf, r
#         return np.log(vol), r
#     except:
#         return -np.inf, r

def estimate_H_by_max_volume(hull_pts, K, verbose=False):
    """
    Returns:
        - H_hat_best: subset of K rows with max volume
        - best_logvol: max volume in log
    """
    
    n = int(hull_pts.shape[0])
    total = int(comb(n, K, exact=True))
    if verbose:
        print(f"H candidates={n}, K={K}, combinations={total:,}")

    # it with optional tqdm progress bar
    it = combinations(range(n), K)
    if verbose:
        try:
            from tqdm import tqdm as _tqdm  # lazy, optional
            it = _tqdm(it, total=total, desc=f"Searching K={K} subsets for max volume")
        except Exception:
            pass  # tqdm not installed; just iterate

    best_logvol = -np.inf
    best_inds = None
    
    for inds in it:
        subset = hull_pts[list(inds)]
        logvol, _ = log_intrinsic_volume_score(subset)
        if logvol > best_logvol:
            best_logvol = logvol
            best_inds = inds

    H_hat_best = hull_pts[list(best_inds)]
    return H_hat_best, float(best_logvol)

def get_affine_basis_trimmed(Y, tol=1e-12):
    Y_reduced = Y[:, :-1]
    mean = Y_reduced.mean(axis=0)
    Y_centered = Y_reduced - mean
    U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    rank = np.sum(S > tol)
    basis = Vt[:rank].T  # (J-1) x rank
    return basis, rank, mean

def project_to_intrinsic(Y_reduced, basis, mean):
    return (Y_reduced - mean) @ basis

def project_to_simplex(v):
    """Project a single vector v onto the probability simplex."""
    v = np.asarray(v)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def compute_C(mu, H):
    numerator = mu[:, None] * H
    denominator = numerator.sum(axis=0)
    C = numerator / denominator
    return C.T

def solve_H_right_inverse(
    Y, H, tol_clip=1e-12, major_tol=1e-2,
    verbose=False, auto_orient=True, renormalize_rows=False
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
    
    # compute diagnostics
    G = H_aug @ H_aug.T
    try:
        condG = float(np.linalg.cond(G))
    except np.linalg.LinAlgError:
        condG = float("inf")
    I_err = float(np.linalg.norm(H_aug @ H_aug_R - np.eye(K), ord=np.inf))
    aug_resid = float(np.linalg.norm(Y_aug - W_raw @ H_aug, ord=np.inf))

    # print diagnostics
    if verbose:
        print(f"||H_aug H_aug_R - I||_inf: {I_err:.3e}")
#         print(f"{major_count} of {n} rows had major simplex violations before clipping")
#         print(f"H transposed: {transposed}")
#         print(f"max row sum dev of H: {np.max(np.abs(H_in.sum(axis=1)-1.0)):.3e}")
#         print(f"rank of H_aug: {np.linalg.matrix_rank(H_aug)} of {K}")
#         print(f"cond number of G: {condG:.3e}")      
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

def prune_close_points(points, K=None, min_K=3, seed=123):
    """
    Clustering based pruning with a user specified minimum K.
    Picks one representative per cluster that is an actual data point.
    """
    X = np.asarray(points, dtype=float)
    # Drop rows with any NaN/inf to avoid sklearn metric issues
    good = np.isfinite(X).all(axis=1)
    X = X[good]
    n = X.shape[0]

    if n == 0:
        return X, []
    if n == 1:
        return X.copy(), [0]

    # Try to import sklearn lazily; if missing, skip pruning safely.
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score
    except Exception:
        # no sklearn: keep all points
        return X, list(range(n))
    
    if K is None:
        lower = max(2, int(min_K))
        upper = min(max(25, lower), n - 1)
        candidates = list(range(lower, upper + 1))
        best_k, best_s = None, -np.inf
        for k in candidates:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=4096)
            labels = km.fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            s = silhouette_score(X, labels, metric="euclidean", sample_size=min(10000, n), random_state=seed)
            if s > best_s:
                best_s, best_k = s, k
        K = best_k if best_k is not None else 1
    else:
        K = int(max(K, min_K))
        if K >= n:
            K = n - 1 if n > 1 else 1
        if K < 1:
            K = 1

    if K == 1:
        return X[[0]], [0]

    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=10, batch_size=4096)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    selected = []
    for k in range(K):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            continue
        diffs = X[members] - centers[k]
        idx = members[np.argmin(np.einsum("ij,ij->i", diffs, diffs))]
        selected.append(idx)

    selected = sorted(set(selected))
    return X[selected], selected

def sourceXray(Y, K, seed=123, prune=False, min_K=None, tol=1e-12, verbose=False):
    """
    For each of the top-10 candidate H_star_hat (by log-volume), estimates:
        (W_star_hat, W_tilde_hat, mu_tilde_hat, C_hat)
    Returns list of tuples: (H_star_hat, W_star_hat, W_tilde_hat, mu_tilde_hat, C_hat, log_volume)
    """
    verbose_flag = verbose
    
    rng = np.random.default_rng(seed)
    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()
    n = Y.shape[0]

    # compute row sums as column vector
    r = Y.sum(axis=1, keepdims=True)

    # normalized versions
    Y_star = Y / r

    # Step 1: Prepare projected hull points if candidates not given
    Y_star_np = Y_star
    Y_star_reduced = Y_star_np[:, :-1]
    basis, rank, mean = get_affine_basis_trimmed(Y_star_np, tol=1e-10)
    Y_star_proj = project_to_intrinsic(Y_star_reduced, basis, mean)   

    if verbose: 
        print("Computing convex hull...", end="", flush=True)
        
    start = time.time()
    hull = ConvexHull(Y_star_proj, qhull_options="Qx Qt Q12 Pp")
    hull_inds = hull.vertices
    
    if verbose:
        print(f" done in {time.time()-start:.2f}s")
 
    hull_pts = Y_star_proj[hull_inds]  # projected coordinates
    hull_pts_ambient = Y_star_np[hull_inds]

    if prune:
        min_K = 10*K if min_K is None else min_K
        pruned_pts, pruned_inds_local = prune_close_points(hull_pts, min_K=min_K)
        pruned_pts_ambient = hull_pts_ambient[pruned_inds_local]
        if verbose: 
            print("Number of pruned hull points:", len(pruned_pts))
        H_candidates = pruned_pts_ambient       
    else:
        H_candidates = hull_pts_ambient

    # Step 2: Get H
    H_star_hat, logvol_hat = estimate_H_by_max_volume(H_candidates, K, verbose=verbose_flag)
    
    results = []

    # Step 3: Get W
    W_star_hat, _, _ = solve_H_right_inverse(Y_star, H_star_hat, verbose=verbose_flag)
    W_tilde_hat = W_star_hat * r 
    mu_tilde_hat = W_tilde_hat.mean(axis=0)
    C_hat = compute_C(mu_tilde_hat, H_star_hat)

    results.append((H_star_hat, W_tilde_hat, mu_tilde_hat, C_hat, logvol_hat))

    return results
