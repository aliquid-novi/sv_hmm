import arviz as az
import pandas as pd
import numpy as np
import prior_pert_funcs as ppf
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan
from pathlib import Path 

BASE_PRIORS = {
    "mu_s":      {"mu": 0.0,              "sigma": 2.0},
    "phi_s_raw": {"mu": np.arctanh(0.98), "sigma": 0.3},
    "kappa_s":   {"mu": np.log(0.10**2),  "sigma": 1.0},

    "mu_f":      {"mu": -1.0,             "sigma": 2.5},
    "phi_f_raw": {"mu": np.arctanh(0.6),  "sigma": 0.5},
    "kappa_f":   {"mu": np.log(0.30**2),  "sigma": 1.0},

    "nu_minus_two": {"rate": 1.0},
}

idata_data = [['USDJPY_idata.json', 'USDJPY.csv'], ['GBPJPY_data.json', 'GBPJPY.csv']]

sample_kwargs = dict(
    draws=15, tune=15,
    target_accept=0.97,
    chains=4, cores=4,
    max_treedepth=12,
    idata_kwargs={"log_likelihood": True},
)

for file in idata_data:
    name, csv_file = file[0], file[1]
    cleaned_name, ext = name.split(".")

    print(f"Iterating through {cleaned_name}")
    idata = az.from_json(file[0])
    data = Path(file[1])
    df = pd.read_csv(data, skipfooter = 1, engine = 'python').set_index('Time')
    df = df.iloc[::-1]
    df["log_ret_diff"] = np.log(df["Latest"]).diff()
    df['pct_change'] = df['Latest'].pct_change()
    df = df.dropna()
    df_sv = df.iloc[-500:]
    y = df_sv["log_ret_diff"].values

    y_used, obs_name = ppf.get_y_used_from_idata(idata)
    ll = ppf.get_ll_matrix(idata, obs_name)
    S, T_ll = ll.shape

    phi_s = ppf.stack_1d(idata, "phi_s")
    phi_f = ppf.stack_1d(idata, "phi_f")
    sigma_s = ppf.stack_1d(idata, "sigma_s")
    sigma_f = ppf.stack_1d(idata, "sigma_f")
    nu_m2 = ppf.stack_1d(idata, "nu_minus_two")

    # baseline means
    base = {
        "phi_s": phi_s.mean(),
        "phi_f": phi_f.mean(),
        "sigma_s": sigma_s.mean(),
        "sigma_f": sigma_f.mean(),
        "nu_minus_two": nu_m2.mean(),
    }

    rows = []
    
    screen_idx = np.argsort(np.abs(y_used))[-50:]

    for t in screen_idx:
        w = ppf.weights_case_deletion_from_ll(ll, t)
        ess = ppf.weight_ess(w)

        # reweighted means (approx posterior if r_t removed)
        row = {
            "t": int(t),
            "ESS_w": float(ess),
            "d_phi_s": float(ppf.weighted_mean(phi_s, w) - base["phi_s"]),
            "d_phi_f": float(ppf.weighted_mean(phi_f, w) - base["phi_f"]),
            "d_sigma_s": float(ppf.weighted_mean(sigma_s, w) - base["sigma_s"]),
            "d_sigma_f": float(ppf.weighted_mean(sigma_f, w) - base["sigma_f"]),
            "d_nu_m2": float(ppf.weighted_mean(nu_m2, w) - base["nu_minus_two"]),
        }
        rows.append(row)

    # sort by influence (you choose a score)
    # e.g., max abs change across key params:
    for r in rows:
        r["score"] = max(abs(r["d_phi_s"]), abs(r["d_phi_f"]), abs(r["d_sigma_s"]), abs(r["d_sigma_f"]))

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)

    sd = np.std(y)
    delta_list = [0.5*sd, -0.5*sd, 1.0*sd, -1.0*sd]

    t1, t2, t3 = rows_sorted[0:3]
    t_list = [t1['t'], t2['t'], t3['t']] 

    # Case Deletion Runs 

    print("Implementing case deletion pertubations...")
    case_deletion_runs = ppf.refit_case_deletion_grid(y, t_list, BASE_PRIORS, cleaned_name)
    
    print("Implementing observation pertubations...")
    obs_shift_runs = ppf.refit_obs_shift_grid(y, cleaned_name, t_list=t_list, delta_list=delta_list, )
    
    print("Implementing prior pertubations...")
    etas = [-0.5, -0.25, 0.0, 0.25, 0.5]
    prior_runs_phi = ppf.refit_prior_perturbation_grid(y, etas, cleaned_name, scheme="phi_raw_scale", prefix="prior_pert")

