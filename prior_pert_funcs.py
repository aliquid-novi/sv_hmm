import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan
import arviz as az

sample_kwargs = dict(
    draws=1500, tune=1500,
    target_accept=0.97,
    chains=4, cores=4,
    max_treedepth=12,
    idata_kwargs={"log_likelihood": True},
)

BASE_PRIORS = {
    "mu_s":      {"mu": 0.0,              "sigma": 2.0},
    "phi_s_raw": {"mu": np.arctanh(0.98), "sigma": 0.3},
    "kappa_s":   {"mu": np.log(0.10**2),  "sigma": 1.0},

    "mu_f":      {"mu": -1.0,             "sigma": 2.5},
    "phi_f_raw": {"mu": np.arctanh(0.6),  "sigma": 0.5},
    "kappa_f":   {"mu": np.log(0.30**2),  "sigma": 1.0},

    "nu_minus_two": {"rate": 1.0},
}

def stack_1d(idata, var):
    """Stack chain/draw -> sample for scalar RVs."""
    return np.asarray(idata.posterior[var].stack(sample=("chain","draw")))

def stack_h(idata, var="h"):
    """Stack chain/draw -> sample for h paths (S,T)."""
    h_da = idata.posterior[var].stack(sample=("chain","draw"))
    # your time dim is likely "h_dim_0"; handle generically:
    time_dim = [d for d in h_da.dims if d != "sample"][0]
    return np.asarray(h_da.transpose("sample", time_dim))

def get_ll_matrix(idata, obs_name="r"):
    """Return loglik matrix shape (S,T)."""
    ll_da = idata.log_likelihood[obs_name].stack(sample=("chain","draw"))
    time_dim = [d for d in ll_da.dims if d != "sample"][0]
    ll = np.asarray(ll_da.transpose("sample", time_dim))
    return ll

def weights_case_deletion_from_ll(ll, t):
    """ll: (S,T). w ∝ exp(-ll[:,t])."""
    logw = -ll[:, t]
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= w.sum()
    return w

def weight_ess(w):
    w = np.asarray(w)
    w = w / w.sum()
    return 1.0 / np.sum(w**2)

def weighted_mean(x, w):
    return np.sum(x * w)

def weighted_mean_path(X, w):
    """X: (S,T)"""
    return (X * w[:, None]).sum(axis=0)

def _clip_pos(x, eps=1e-8):
    return float(max(x, eps))

def perturb_normal_scale(mu, sigma, eta):
    # sigma' = sigma * exp(eta)
    return mu, _clip_pos(sigma * np.exp(eta))

def perturb_exponential_rate(rate, eta):
    # rate' = rate * exp(eta)
    return _clip_pos(rate * np.exp(eta))

def obs_shift(y, t, delta):
    y = np.asarray(y).copy()
    y[t] = y[t] + delta
    return y

def get_y_used_from_idata(idata, obs_name=None):
    if obs_name is None:
        obs_name = list(idata.observed_data.data_vars)[0]  # e.g. "r"
    y_used = np.asarray(idata.observed_data[obs_name])
    return y_used, obs_name

def get_ll_matrix(idata, obs_name):
    ll_da = idata.log_likelihood[obs_name].stack(sample=("chain","draw"))
    time_dim = [d for d in ll_da.dims if d != "sample"][0]
    ll = np.asarray(ll_da.transpose("sample", time_dim))
    return ll  # (S, T)

### grids 

def refit_case_deletion_grid(y, t_list, priors, prefix = "case_deletion"):
    
    out = {}
    
    for t in t_list:
        
        y_case_deletion = np.delete(y, t)
        m = build_sv_model(y_case_deletion, priors=priors)
        with m:
            idata = pm.sample(**sample_kwargs)
        key = (f'{t}_case_deletion') 
        out[key] = idata
        az.to_json(idata, f"{prefix}_t{t}_deletion.json")
    
    return out

def refit_obs_shift_grid(y, t_list, delta_list, priors=BASE_PRIORS, prefix="obs_shift"):
    out = {}
    for t in t_list:
        for delta in delta_list:
            y_pert = obs_shift(y, t, delta)
            m = build_sv_model(y_pert, priors=priors)
            with m:
                idata = pm.sample(**sample_kwargs)
            key = (int(t), float(delta))
            out[key] = idata
            az.to_json(idata, f"{prefix}_t{t}_d{delta:+.4f}.json")
    return out

def apply_prior_perturbation(base, eta, scheme="scale_all"):
    p = {k: v.copy() for k, v in base.items()}

    if scheme == "scale_all":
        for k in ["mu_s","phi_s_raw","kappa_s","mu_f","phi_f_raw","kappa_f"]:
            mu, sig = p[k]["mu"], p[k]["sigma"]
            mu2, sig2 = perturb_normal_scale(mu, sig, eta)
            p[k]["mu"], p[k]["sigma"] = mu2, sig2

        p["nu_minus_two"]["rate"] = perturb_exponential_rate(p["nu_minus_two"]["rate"], eta)

    elif scheme == "phi_raw_scale":
        for k in ["phi_s_raw","phi_f_raw"]:
            mu, sig = p[k]["mu"], p[k]["sigma"]
            mu2, sig2 = perturb_normal_scale(mu, sig, eta)
            p[k]["mu"], p[k]["sigma"] = mu2, sig2

    elif scheme == "kappa_scale":
        for k in ["kappa_s","kappa_f"]:
            mu, sig = p[k]["mu"], p[k]["sigma"]
            mu2, sig2 = perturb_normal_scale(mu, sig, eta)
            p[k]["mu"], p[k]["sigma"] = mu2, sig2

    elif scheme == "tails":
        p["nu_minus_two"]["rate"] = perturb_exponential_rate(p["nu_minus_two"]["rate"], eta)

    else:
        raise ValueError("unknown scheme")

    return p

    ### Prior Pertubation Run 
def refit_prior_perturbation_grid(y, etas, scheme="scale_all", prefix="prior_pert"):
    out = {}
    for eta in etas:
        pri_eta = apply_prior_perturbation(BASE_PRIORS, eta=eta, scheme=scheme)
        m = build_sv_model(y, priors=pri_eta)
        with m:
            idata = pm.sample(**sample_kwargs)
        out[float(eta)] = (idata, pri_eta)
        az.to_json(idata, f"{prefix}_{scheme}_eta{eta:+.3f}.json")
    return out

etas = [-0.5, -0.25, 0.0, 0.25, 0.5]


### SV Model

def ar1_build(eps, h0, mu, phi, sigma):
    def step(eps_t, h_prev, mu, phi, sigma):
        return mu + phi * (h_prev - mu) + sigma * eps_t
    h_tail, _ = scan(fn=step, sequences=[eps], outputs_info=[h0], non_sequences=[mu, phi, sigma])
    return pt.concatenate([[h0], h_tail])

def build_sv_model(y, priors=BASE_PRIORS):
    y = np.asarray(y)
    T = len(y)

    with pm.Model() as m:
        # slow priors
        mu_s = pm.Normal("mu_s", priors["mu_s"]["mu"], priors["mu_s"]["sigma"])
        phi_s_raw = pm.Normal("phi_s_raw", priors["phi_s_raw"]["mu"], priors["phi_s_raw"]["sigma"])
        phi_s = pm.Deterministic("phi_s", pt.tanh(phi_s_raw))
        kappa_s = pm.Normal("kappa_s", priors["kappa_s"]["mu"], priors["kappa_s"]["sigma"])
        sigma_s = pm.Deterministic("sigma_s", pt.exp(0.5 * kappa_s))

        # fast priors
        mu_f = pm.Normal("mu_f", priors["mu_f"]["mu"], priors["mu_f"]["sigma"])
        phi_f_raw = pm.Normal("phi_f_raw", priors["phi_f_raw"]["mu"], priors["phi_f_raw"]["sigma"])
        phi_f = pm.Deterministic("phi_f", pt.tanh(phi_f_raw))
        kappa_f = pm.Normal("kappa_f", priors["kappa_f"]["mu"], priors["kappa_f"]["sigma"])
        sigma_f = pm.Deterministic("sigma_f", pt.exp(0.5 * kappa_f))

        # initial states
        h0_s = pm.Normal("h0_s", mu=mu_s, sigma=sigma_s / pt.sqrt(1 - phi_s**2 + 1e-12))
        h0_f = pm.Normal("h0_f", mu=mu_f, sigma=sigma_f / pt.sqrt(1 - phi_f**2 + 1e-12))

        # non-centred innovations
        eps_s = pm.Normal("eps_s", 0.0, 1.0, shape=T-1)
        eps_f = pm.Normal("eps_f", 0.0, 1.0, shape=T-1)

        h_s = pm.Deterministic("h_s", ar1_build(eps_s, h0_s, mu_s, phi_s, sigma_s))
        h_f = pm.Deterministic("h_f", ar1_build(eps_f, h0_f, mu_f, phi_f, sigma_f))
        h = pm.Deterministic("h", h_s + h_f)

        nu_minus_two = pm.Exponential("nu_minus_two", priors["nu_minus_two"]["rate"])
        nu = pm.Deterministic("nu", nu_minus_two + 2.0)

        pm.StudentT("r", nu=nu, mu=0.0, sigma=pt.exp(h/2), observed=y)


    return m
