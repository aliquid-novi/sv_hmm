import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pathlib import Path
from pytensor.scan import scan

data_file = ['LYC_Daily.csv', '4D_Daily.csv', 'CBA_Daily.csv', 'EURUSD.csv', 'GBPJPY.csv', 'USDJPY.csv']

def ar1_build(eps, h0, mu, phi, sigma):
    def step(eps_t, h_prev, mu, phi, sigma):
        return mu + phi * (h_prev - mu) + sigma * eps_t

    h_tail = scan(
        fn=step,
        sequences=[eps],
        outputs_info=[h0],
        non_sequences=[mu, phi, sigma],
        return_updates=False,  # <-- add this
    )

    return pt.concatenate([[h0], h_tail])  # length T
        
for data in data_file:

    data = Path(data)
    df = pd.read_csv(data, skipfooter = 1, engine = 'python').set_index('Time')
    df = df.iloc[::-1]
    df["log_ret_diff"] = np.log(df["Latest"]).diff()
    df['pct_change'] = df['Latest'].pct_change()
    df = df.dropna()
    df.head()

    df_sv = df.iloc[-1250:]
    
    with pm.Model() as m:
        y = df_sv["log_ret_diff"].values
        T = len(y)
    
        # --- slow parameters ---
        mu_s = pm.Normal("mu_s", 0.0, 2.0)
        phi_s_raw = pm.Normal("phi_s_raw", mu=np.arctanh(0.98), sigma=0.3)
        phi_s = pm.Deterministic("phi_s", pt.tanh(phi_s_raw))
        kappa_s = pm.Normal("kappa_s", mu=np.log(0.10**2), sigma=1.0)     # log variance
        sigma_s = pm.Deterministic("sigma_s", pt.exp(0.5 * kappa_s))
    
        # --- fast parameters (ANCHOR mean to avoid ridge) ---
        mu_f = pm.Normal("mu_f", -1.0, 2.5)
        phi_f_raw = pm.Normal("phi_f_raw", mu=np.arctanh(0.6), sigma=0.5)
        phi_f = pm.Deterministic("phi_f", pt.tanh(phi_f_raw))
        kappa_f = pm.Normal("kappa_f", mu=np.log(0.30**2), sigma=1.0)
        sigma_f = pm.Deterministic("sigma_f", pt.exp(0.5 * kappa_f))
    
        # --- initial states (stationary-ish) ---
        h0_s = pm.Normal("h0_s", mu=mu_s, sigma=sigma_s / pt.sqrt(1 - phi_s**2 + 1e-12))
        h0_f = pm.Normal("h0_f", mu=mu_f, sigma=sigma_f / pt.sqrt(1 - phi_f**2 + 1e-12))
    
        # --- non-centred innovations ---
        eps_s = pm.Normal("eps_s", 0.0, 1.0, shape=T-1)
        eps_f = pm.Normal("eps_f", 0.0, 1.0, shape=T-1)
    
        h_s = pm.Deterministic("h_s", ar1_build(eps_s, h0_s, mu_s, phi_s, sigma_s))
        h_f = pm.Deterministic("h_f", ar1_build(eps_f, h0_f, mu_f, phi_f, sigma_f))
        h = pm.Deterministic("h", h_s + h_f)
    
        # observation (Student-t, as you changed)
        nu = pm.Exponential("nu_minus_two", 1.0) + 2
        pm.StudentT("r", nu=nu, mu=0.0, sigma=pt.exp(h/2), observed=y)

        print(f"Now running model for {data}...")
        
        idata = pm.sample(30, tune = 30, target_accept=0.97, chains=4, cores = 4, max_treedepth = 12)     

        az.to_json(idata, f'{data}_idata.json')
        print(f"Exported {data}")
