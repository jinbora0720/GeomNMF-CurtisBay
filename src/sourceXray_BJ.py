import time
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.special import gammaln  # log-factorial
from scipy.special import comb
from src.NFINDR import nfindr_BJ

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

# def get_affine_basis_trimmed(Y, tol=1e-12):
#     Y_reduced = Y[:, :-1]
#     mean = Y_reduced.mean(axis=0)
#     Y_centered = Y_reduced - mean
#     U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
#     rank = np.sum(S > tol)
#     basis = Vt[:rank].T  # (J-1) x rank
#     return basis, rank, mean

# def project_to_intrinsic(Y_reduced, basis, mean):
#     return (Y_reduced - mean) @ basis

def _candidates_random_directions(Yc_proj_w, seed=123, 
                                  T=20000, # how many random directions to sample
                                  topk=1, # how many extremes to pick per direction
                                  max_K=None, # how many candidates to return based on counts (importance)
                                  verbose=False):
    """
    Returns candidate indices sorted by frequency (descending).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(Yc_proj_w, float)
    n, r = X.shape

    counts = np.zeros(n, dtype=int)

    if verbose:
        print(f"Sampling {topk} extreme points in each of {T} random directions...", end="", flush=True)
    start = time.time()

    for _ in range(T):
        u = rng.normal(size=r) # random direction
        u /= np.linalg.norm(u) + 1e-12
        s = X @ u # score along random direction

        if topk == 1:
            idxs = [int(np.argmax(s))]
        else:
            idxs = np.argpartition(s, -topk)[-topk:]
            idxs = idxs[np.argsort(s[idxs])[::-1]] # sort topk in descending order

        for idx in idxs:
            counts[idx] += 1

        # opposite direction
        if topk == 1:
            idxs2 = [int(np.argmin(s))]
        else:
            idxs2 = np.argpartition(s, topk)[:topk]
            idxs2 = idxs2[np.argsort(s[idxs2])]
        for idx in idxs2:
            counts[idx] += 1

    chosen = np.flatnonzero(counts > 0)
    chosen_sorted = chosen[np.argsort(counts[chosen])[::-1]] # min(n, 2T x topk) candidates, sorted by count

    if max_K is not None:
        chosen_sorted = chosen_sorted[:max_K] # min(n, 2T x topk, max_K) candidates

    if verbose:
        print(f" done in {time.time()-start:.2f}s; #cands={len(chosen_sorted)}")
    return chosen_sorted, counts

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a single vector *v* onto the probability simplex.

    Uses the O(n log n) sort-based algorithm of Duchi et al. (2008).

    Parameters
    ----------
    v : np.ndarray, shape (n,)
        Input vector to project.

    Returns
    -------
    w : np.ndarray, shape (n,)
        Projection of *v* onto the unit probability simplex
        ``{w ≥ 0, sum(w) = 1}``.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    u = np.sort(v)[::-1] # v sorted in descending order
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)

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

# def sourceXray(Y, K, seed=123, prune=False, min_K=None, tol=1e-12, verbose=False):
#     """
#     For each of the top-10 candidate H_star_hat (by log-volume), estimates:
#         (W_star_hat, W_tilde_hat, mu_tilde_hat, C_hat)
#     Returns list of tuples: (H_star_hat, W_star_hat, W_tilde_hat, mu_tilde_hat, C_hat, log_volume)
#     """
#     verbose_flag = verbose
    
#     rng = np.random.default_rng(seed)
#     if isinstance(Y, pd.DataFrame):
#         Y = Y.to_numpy()
#     n = Y.shape[0]

#     # compute row sums as column vector
#     r = Y.sum(axis=1, keepdims=True)

#     # normalized versions
#     Y_star = Y / r

#     # Step 1: Prepare projected hull points if candidates not given
#     Y_star_np = Y_star
#     Y_star_reduced = Y_star_np[:, :-1]
#     basis, rank, mean = get_affine_basis_trimmed(Y_star_np, tol=tol)
#     Y_star_proj = project_to_intrinsic(Y_star_reduced, basis, mean)   

#     if verbose: 
#         print("Computing convex hull...", end="", flush=True)
        
#     start = time.time()
#     hull = ConvexHull(Y_star_proj, qhull_options="Qx Qt Q12 Pp")
#     hull_inds = hull.vertices
    
#     if verbose:
#         print(f" done in {time.time()-start:.2f}s")
 
#     hull_pts = Y_star_proj[hull_inds]  # projected coordinates
#     hull_pts_ambient = Y_star_np[hull_inds]

#     if prune:
#         min_K = 10*K if min_K is None else min_K
#         pruned_pts, pruned_inds_local = prune_close_points(hull_pts, min_K=min_K)
#         pruned_pts_ambient = hull_pts_ambient[pruned_inds_local]
#         if verbose: 
#             print("Number of pruned hull points:", len(pruned_pts))
#         H_candidates = pruned_pts_ambient       
#     else:
#         H_candidates = hull_pts_ambient

#     # Step 2: Get H
#     H_star_hat, logvol_hat = estimate_H_by_max_volume(H_candidates, K, verbose=verbose_flag)
    
#     results = []

#     # Step 3: Get W
#     W_star_hat, _, _ = solve_H_right_inverse(Y_star, H_star_hat, verbose=verbose_flag)
#     W_tilde_hat = W_star_hat * r 
#     mu_tilde_hat = W_tilde_hat.mean(axis=0)
#     C_hat = compute_C(mu_tilde_hat, H_star_hat)

#     results.append((H_star_hat, W_tilde_hat, mu_tilde_hat, C_hat, logvol_hat))

#     return results

def sourceXray(Y, K, seed=123, tol=1e-12,
               candidate_method="exact", # "random" (random directions)
               T=20000, topk=1, max_K=None, # for random candidate method
               prune=False, min_K=None, # for pruning
               refine_greedy=False, # N-FINDR style refinement 
               return_fit_diagnostics=False, # temporary solution not to damage running scripts but to have a similar functionality as GeomNMF 
               verbose=False):
    """
    For each of the top-10 candidate H_star_hat (by log-volume), estimates:
        (W_star_hat, W_tilde_hat, mu_tilde_hat, C_hat)
    Returns list of tuples: (H_star_hat, W_star_hat, W_tilde_hat, mu_tilde_hat, C_hat, log_volume)
    """
    verbose_flag = verbose

    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()
    n = Y.shape[0]

    # compute row sums as column vector
    r = Y.sum(axis=1, keepdims=True)

    # normalized versions
    Y_star = Y / r

    # Step 0: Project to intrinsic space
    # drop last coordinate to work in R^{J-1} affine embedding
    Y_star_reduced = Y_star[:, :-1].astype(float, copy=False)  # (n, J-1)
    
    # fit affine basis via SVD on centered reduced coordinates
    mean = Y_star_reduced.mean(axis=0, keepdims=True)          # (1, J-1)
    Yc = Y_star_reduced - mean
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    mask = S > tol
    basis = Vt[mask].T

    # project to intrinsic coords
    Yc_proj = Yc @ basis # (n, rank) 
    
    # Step 1: candidates
    counts = None  # only populated for candidate_method="random"
    if candidate_method == "exact":
        if verbose: 
            print("Computing convex hull...", end="", flush=True)
        
        start = time.time()
        hull = ConvexHull(Yc_proj, qhull_options="Qx Qt Q12 Pp")
        cand_idx = hull.vertices

        if verbose:
            print(f" done in {time.time()-start:.2f}s; #cands={len(cand_idx)}")
        
        cand_proj = Yc_proj[cand_idx]  # projected coordinates

    elif candidate_method == "random":
        # whitening: scale to unit covariance (remove directional stretch and make all directions comparable)
        # fit covariance in intrinsic space and build inv sqrt
        C = (Yc_proj.T @ Yc_proj) / max(n - 1, 1)  # (rank, rank)
        evals, evecs = np.linalg.eigh(C)
        evals = np.maximum(evals, tol)
        inv_sqrt_cov = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T  # (rank, rank)
        Yc_proj_w = Yc_proj @ inv_sqrt_cov # (n, rank)

        cand_idx, counts = _candidates_random_directions(Yc_proj_w, seed=seed, T=T, topk=topk, max_K=max_K, verbose=verbose)
        cand_proj = Yc_proj[cand_idx]  

    else:
        raise ValueError("candidate_method must be 'exact' or 'random'")

    cand_ambient = Y_star[cand_idx] 

    # distance based pruning is good to handle "near duplicates"
    if prune:
        min_K = 10*K if min_K is None else min_K
        if min_K >= len(cand_proj):
            if verbose:
                print(f"Pruning skipped since candidate count {len(cand_proj)} <= min_K {min_K}")
        else:
            pruned_pts, pruned_idx_local = prune_close_points(cand_proj, min_K=min_K)
            cand_ambient = cand_ambient[pruned_idx_local]
            cand_idx = cand_idx[pruned_idx_local]
            if verbose: 
                print("Number of pruned candidates:", len(pruned_pts))

    H_candidates = cand_ambient

    # Step 2: Get H
    H_star_hat, logvol_hat = estimate_H_by_max_volume(H_candidates, K, verbose=verbose_flag)
    
    # refine by greedy search
    logvol_before = None
    if refine_greedy:
        if candidate_method == "exact" and not prune:
            if verbose:
                print("Greedy refinement skipped: exact hull with no pruning already guarantees global optimum over candidates.")
        else:
            # save logvol before refinement
            logvol_before = logvol_hat
            # map H_star_hat rows back to indices in Y_star
            diffs = Y_star[:, np.newaxis, :] - H_star_hat[np.newaxis, :, :]  # (n, K, J)
            init_idx = np.argmin(np.sum(diffs**2, axis=2), axis=0)            # (K,)
            
            _, refined_idx = nfindr_BJ(
                Y_star, K,
                normalize=False,
                init='atgp', # ignored since init_idx is not None
                init_idx=init_idx,
                seed=seed,
            )
            H_refined = Y_star[refined_idx]
            logvol_refined, _ = log_intrinsic_volume_score(H_refined) # best_vol from nfindr_BJ is NOT the same vol

            # only accept refinement if it actually improves volume in Y_star space
            if logvol_refined > logvol_before:
                H_star_hat = H_refined
                logvol_hat = logvol_refined
                if verbose:
                    vol_ratio = np.exp(logvol_refined - logvol_before)
                    print(
                            f"Greedy refinement accepted; log-vol "
                            f"{logvol_before:.4f} → {logvol_refined:.4f} "
                            f"(volume ratio: {vol_ratio:.3f}x)"
                        )
            else:
                if verbose:
                    print(
                        f"Greedy refinement rejected; refined log-vol "
                        f"{logvol_refined:.4f} did not improve over "
                        f"{logvol_before:.4f}"
                    )

    results = []

    # Step 3: Get W
    W_star_hat, _, _ = solve_H_right_inverse(Y_star, H_star_hat, verbose=verbose_flag)
    W_tilde_hat = W_star_hat * r 
    mu_tilde_hat = W_tilde_hat.mean(axis=0)
    C_hat = compute_C(mu_tilde_hat, H_star_hat)

    results.append((H_star_hat, W_tilde_hat, mu_tilde_hat, C_hat, logvol_hat))

    if return_fit_diagnostics:
        fit_diagnostics = {
            "intrinsic_rank": int(mask.sum()),
            "n_cand": len(H_candidates),
            "cand_idx": cand_idx,
            "direction_hit_counts": counts,                # np.ndarray if candidate_method="random", else None
            "logvol_before_refinement": logvol_before,     # float if refine_greedy ran, else None
        }
        return results, fit_diagnostics
    return results