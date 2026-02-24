# sv_hmm

This repository contains research code and data for a Bayesian stochastic volatility framework with regime structure, developed as part of my quantitative finance capstone at UTS.

The project implements a modular Python pipeline for modelling latent log-volatility using a first-order autoregressive (AR(1)) process estimated via Hamiltonian Monte Carlo (No-U-Turn Sampler). In addition to baseline inference, the framework incorporates regime analysis and structured robustness testing to evaluate the stability of inferred parameters and volatility paths under data and prior perturbations.

## Overview

Volatility is treated as an unobserved latent state evolving through time, with persistence governed by an AR(1) structure. Posterior inference is performed using NUTS, enabling full distributional estimation of:

- Latent daily log-volatility  
- Volatility persistence parameters  
- Innovation variance  
- Regime-dependent behaviour  

To assess model robustness, the framework introduces perturbation-based stress testing, including:

- Removal and shifting of high-volatility observations  
- Scaling of prior distributions  
- Re-estimation of posterior distributions under perturbed scenarios  

Baseline and perturbed posterior distributions are compared using KL Divergence and ELPD metrics to evaluate structural sensitivity and inferential stability.

## Key Features

- Bayesian stochastic volatility modelling (AR(1) latent structure)
- Regime-aware volatility analysis
- Hamiltonian Monte Carlo (NUTS) inference via PyMC
- Prior and data perturbation framework for robustness testing
- Posterior comparison using KL Divergence and ELPD
- Application to equity (ASX) and FX datasets

## Technical Stack

- Python  
- PyMC  
- NumPy / pandas  
- ArviZ  
- SciPy  

## Research Context

This repository focuses not only on model construction, but on model interrogation. The objective is to understand:

- How volatility persistence behaves across regimes  
- When posterior inference is structurally stable  
- Where modelling assumptions may break under stress  
- How volatility dynamics can inform risk-aware derivatives analysis  

The framework is designed as a foundation for further extensions into volatility risk premia research, regime-transition modelling, and posterior predictive stress simulation for derivatives applications.

## Intended Use

This project is suitable for:

- Research in stochastic volatility and regime-switching models  
- Bayesian time-series modelling  
- Quantitative risk analysis and robustness evaluation  
- Foundations for systematic volatility-based strategy research  

Future extensions may include alternative inference techniques (e.g. particle MCMC, Kalman-based approaches), explicit regime-switching state processes, and integration with option pricing and risk metrics (VaR / Expected Shortfall).

---
