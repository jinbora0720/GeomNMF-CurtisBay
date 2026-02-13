# Import 
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
from itertools import permutations
import time
import pandas as pd
from joblib import dump, load
from pathlib import Path

## sourceXray
from src.sourceXray_BJ import sourceXray, compute_C, solve_H_right_inverse
from src.utils import permute_estimates_to_match_truth, quantiles_by_group

## X-RAY
from src.XRAY import xray_BJ
#-----------------------------------------------------------------------------------------------------------------------------#

# Load data
obj = load("data/CB_10locs_complete_nonegpm25mpm1.joblib")
df = obj["df"]
ys = ['pm1', 'pm25mpm1', 'pm10mpm25', 'tspmpm10', 'bc', 'co', 'no', 'no2']
Y_org = df.loc[:, ys]
col_label = ["PM1", "PM2.5-PM1", "PM10-PM2.5", "TSP-PM10", "BC", "CO", "NO", "NO2"]
#-----------------------------------------------------------------------------------------------------------------------------#

# path 
outdir = Path("results/CB")
outdir.mkdir(parents=True, exist_ok=True)

# replicate settings
seed = 1
K = 3
n, J = Y_org.shape
jitter_ranges = {
    "pm1": 0.14,
    "pm25mpm1": 0.11,
    "pm10mpm25": 0.34,
    "tspmpm10": 0.34,
    "bc": 0.05,
    "co": 0.20,
    "no": 0.20,
    "no2": 0.35,
}

# reference H 
results = load(outdir/"sourceXray_CB_nonegpm25mpm1_K3.joblib")
H_star = results[40*K]["H_star"]

# create covariates 
b = pd.to_numeric(df['bulldozer'], errors='coerce')
d = pd.to_numeric(df['downwind'],  errors='coerce')
df['bulldozerxdownwind'] = (b*d).astype('Int8')
groups = [0, 1] # ignore NA
G = len(groups)

# select quantiles 
q_levels = np.array([0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 1], dtype=float)  
Q = len(q_levels)

# allocate save arrays
results_reps = {
    "seed": np.nan,
    
    "logvol_sourceXray": np.nan, 
    
    "time_sourceXray": np.nan,
    "time_xray": np.nan,
    
    "C_hat_sourceXray": np.empty((J, K), dtype=float),
    "C_hat_xray": np.empty((J, K), dtype=float),

    "mu_tilde_hat_sourceXray": np.empty((K,), dtype=float),
    
    # to recover W later
    "H_star_hat_sourceXray": np.empty((K, J), dtype=float),
    
    # quantile by group
    "W_tilde_qbybulldozer_sourceXray": np.full((G, Q, K), np.nan, float), 
    "W_tilde_qbydownwind_sourceXray": np.full((G, Q, K), np.nan, float), 
    "W_tilde_qbyinteraction_sourceXray": np.full((G, Q, K), np.nan, float),
    "W_tilde_qbybulldozer_xray": np.full((G, Q, K), np.nan, float), 
    "W_tilde_qbydownwind_xray": np.full((G, Q, K), np.nan, float), 
    "W_tilde_qbyinteraction_xray": np.full((G, Q, K), np.nan, float),
}
#-----------------------------------------------------------------------------------------------------------------------------#

# replicate 
rep_env = os.environ.get("SLURM_ARRAY_TASK_ID")
rep = int(rep_env) if rep_env else 1
seed_rep = seed+rep
rng = np.random.default_rng(seed_rep)
results_reps["seed"] = seed_rep

# jitter Y
Y_jittered = Y_org.copy()
for col, a in jitter_ranges.items():
    u = rng.uniform(-a, a, size=n)          # runif(n, -a, a)
    Y_jittered[col] = (1.0 + u) * Y_jittered[col]

# normalize
Y = Y_jittered.copy()
for col in Y.columns:
    iqnr_j = np.diff(np.quantile(Y[col], [0, 0.85], axis=0))
    min_j = Y[col].min()
    if min_j < 0: 
        Y[col] = Y[col] - min_j
    Y[col] = Y[col]/iqnr_j

Y = np.asarray(Y)
r = Y.sum(axis=1, keepdims=True)
Y_star = Y / r
bulldozer = df['bulldozer'].to_numpy()                   # covariate aligned with resampled rows
downwind = df['downwind'].to_numpy()
interaction = df['bulldozerxdownwind'].to_numpy()

# sourceXray
start = time.time()
H_star_hat, W_tilde_hat, mu_tilde_hat, C_hat, logvol_hat = sourceXray(Y, K, seed=seed_rep, prune=True, min_K=40*K, verbose=False)[0]
end = time.time()
results_reps["time_sourceXray"] = end - start
results_reps["logvol_sourceXray"] = logvol_hat

## permute
H_star_hat_perm, mu_tilde_hat_perm, C_hat_perm, order = permute_estimates_to_match_truth(H_star, H_star_hat, mu_tilde_hat, C_hat)
results_reps["mu_tilde_hat_sourceXray"] = np.asarray(mu_tilde_hat_perm)
results_reps["C_hat_sourceXray"] = np.asarray(C_hat_perm)
results_reps["H_star_hat_sourceXray"] = np.asarray(H_star_hat_perm)
W_tilde_hat_perm = W_tilde_hat[:,order]

## quantiles of W tilde by covariates
results_reps["W_tilde_qbybulldozer_sourceXray"] = quantiles_by_group(W_tilde_hat_perm, bulldozer, q_levels)
results_reps["W_tilde_qbydownwind_sourceXray"] = quantiles_by_group(W_tilde_hat_perm, downwind, q_levels)
results_reps["W_tilde_qbyinteraction_sourceXray"] = quantiles_by_group(W_tilde_hat_perm, interaction, q_levels)

# X-RAY
start = time.time()
H_star_hat_xray, _ = xray_BJ(Y_star, K, seed = seed_rep, normalize = False, prefer_scipy_nnls = True)
W_star_hat_xray, _, _ = solve_H_right_inverse(Y_star, H_star_hat_xray)
W_rs = W_star_hat_xray.sum(axis=1, keepdims=True)
W_star_hat_xray = W_star_hat_xray/W_rs
W_tilde_hat_xray = W_star_hat_xray * r     
mu_tilde_hat_xray = W_tilde_hat_xray.mean(axis=0)
C_hat_xray = compute_C(mu_tilde_hat_xray, H_star_hat_xray)
end = time.time()
results_reps["time_xray"] = end - start

## permute
H_star_hat_perm_xray, mu_tilde_hat_perm_xray, C_hat_perm_xray, order = permute_estimates_to_match_truth(H_star, H_star_hat_xray, mu_tilde_hat_xray, C_hat_xray)
results_reps["C_hat_xray"] = np.asarray(C_hat_perm_xray)
W_tilde_hat_perm_xray = W_tilde_hat_xray[:,order]

## quantiles of W tilde by covariates
results_reps["W_tilde_qbybulldozer_xray"] = quantiles_by_group(W_tilde_hat_perm_xray, bulldozer, q_levels)
results_reps["W_tilde_qbydownwind_xray"] = quantiles_by_group(W_tilde_hat_perm_xray, downwind, q_levels)
results_reps["W_tilde_qbyinteraction_xray"] = quantiles_by_group(W_tilde_hat_perm_xray, interaction, q_levels)

dump(results_reps, outdir/"wrandomerror"/f"CB_nonegpm25mpm1_K3_wrandomerror_rep{rep}.joblib")   

