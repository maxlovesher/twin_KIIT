"""Glucose–insulin physiology shared by the ground-truth simulator and the patient twin.

Model (per patient, time in minutes):

    dG/dt  = -(SG + X·m)·G + SG·Gb + Ra(t) + EGP(t) + dawn·d(h) - k_ex·e(t)·G - k_ren·(G-180)+
    dX/dt  = -p2·X + p2·SI·(I + I_exo)
    dI/dt  = -n·I + β·(G - Gb)+ + s0
    gut    : two-compartment absorption, fast (high-GI) and slow (low-GI) channels
    m      = si_mult(t) · (1 + 0.35·exercise_memory)

It is a Bergman minimal model extended with endogenous secretion (β), gut absorption,
a two-compartment subcutaneous insulin depot for injected doses,
a dawn phenomenon, exercise and medication effects (metformin → Gb/SI, sulfonylurea → s0,
DPP-4i → β, SGLT2i → k_ren, basal insulin → I_exo).

The ground-truth simulator drives this model with *hidden* modulations (stress, sleep debt,
unlogged meals). The digital twin uses the same structure but only sees what a real system
would see: logged meals, steps and CGM.
"""
from __future__ import annotations

import math

import numpy as np

try:  # numba makes the per-minute loops ~100x faster; fall back to pure Python if absent
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

# Parameter vector layout -------------------------------------------------------------
P_GB, P_SG, P_SI, P_P2, P_N, P_BETA, P_S0, P_IEXO, P_KF, P_KS, P_VG, P_DAWN, P_KEX, P_KREN = range(14)
N_PARAMS = 14
PARAM_NAMES = ["Gb", "SG", "SI", "p2", "n", "beta", "s0", "I_exo",
               "k_fast", "k_slow", "Vg", "dawn", "k_ex", "k_ren"]

# State vector layout ------------------------------------------------------------------
S_G, S_X, S_I, S_Q1F, S_Q2F, S_Q1S, S_Q2S, S_EXM, S_SC1, S_SC2 = range(10)
N_STATES = 10

EX_SI_GAIN = 0.5         # max relative insulin-sensitivity boost from recent exercise (post-exercise SI +20-50%)
EX_MEM_TAU = 240.0       # min, decay of the exercise after-effect
CARB_BIOAVAIL = 0.8   # mixed meals incl. splanchnic first-pass extraction
RENAL_THRESHOLD = 180.0  # mg/dL, SGLT2 excretion threshold (simplified)
K_SC = 0.02              # 1/min, absorption rate of injected insulin (two-compartment s.c. depot)
INS_BIOAVAIL = 0.7       # fraction of an s.c. dose reaching plasma
INS_VOL_ML_PER_KG = 120.0  # insulin distribution volume


def default_params(n: int = 1) -> np.ndarray:
    """A typical mid-severity T2D parameter set (n copies, shape (n, N_PARAMS))."""
    p = np.zeros((n, N_PARAMS))
    p[:, P_GB] = 140.0
    p[:, P_SG] = 0.014
    p[:, P_SI] = 2.5e-4
    p[:, P_P2] = 0.03
    p[:, P_N] = 0.14
    p[:, P_BETA] = 0.08
    p[:, P_S0] = 0.0
    p[:, P_IEXO] = 0.0
    p[:, P_KF] = 0.035
    p[:, P_KS] = 0.014
    p[:, P_VG] = 1.9 * 70.0
    p[:, P_DAWN] = 0.15
    p[:, P_KEX] = 0.007
    p[:, P_KREN] = 0.0
    return p


@njit(cache=True)
def dawn_shape(hour: float) -> float:
    """Smooth early-morning hepatic glucose surge, peaking around 06:00."""
    return math.exp(-((hour - 6.0) / 1.5) ** 2)


@njit(cache=True)
def step(s, p, carbs_fast, carbs_slow, ex, si_mult, egp_extra, hour, dt):
    """Advance state ``s`` (modified in place) by ``dt`` minutes with explicit Euler.

    carbs_* are grams ingested during this step; ex is exercise intensity in [0, ~1.2].
    """
    s[S_Q1F] += carbs_fast
    s[S_Q1S] += carbs_slow
    sc1, sc2 = s[S_SC1], s[S_SC2]
    weight = p[P_VG] / 1.9
    ins_ra = K_SC * sc2 * INS_BIOAVAIL * 1e6 / (INS_VOL_ML_PER_KG * weight)  # µU/mL/min from the depot
    G = s[S_G]
    X = s[S_X]
    I = s[S_I]
    kf = p[P_KF]
    ks = p[P_KS]
    ra = CARB_BIOAVAIL * 1000.0 * (kf * s[S_Q2F] + ks * s[S_Q2S]) / p[P_VG]
    m = si_mult * (1.0 + EX_SI_GAIN * s[S_EXM])
    excess = G - RENAL_THRESHOLD
    renal = p[P_KREN] * excess if excess > 0.0 else 0.0
    dG = (-(p[P_SG] + X * m) * G + p[P_SG] * p[P_GB] + ra + egp_extra
          + p[P_DAWN] * dawn_shape(hour) - p[P_KEX] * ex * G - renal)
    dX = -p[P_P2] * X + p[P_P2] * p[P_SI] * (I + p[P_IEXO])
    above = G - p[P_GB]
    dI = -p[P_N] * I + (p[P_BETA] * above if above > 0.0 else 0.0) + p[P_S0] + ins_ra
    q1f, q2f, q1s, q2s = s[S_Q1F], s[S_Q2F], s[S_Q1S], s[S_Q2S]
    s[S_G] = max(G + dt * dG, 20.0)
    s[S_X] = max(X + dt * dX, 0.0)
    s[S_I] = max(I + dt * dI, 0.0)
    s[S_Q1F] = q1f - dt * kf * q1f
    s[S_Q2F] = q2f + dt * (kf * q1f - kf * q2f)
    s[S_Q1S] = q1s - dt * ks * q1s
    s[S_Q2S] = q2s + dt * (ks * q1s - ks * q2s)
    exm = s[S_EXM] + dt * (0.04 * ex - s[S_EXM] / EX_MEM_TAU)
    s[S_EXM] = min(max(exm, 0.0), 1.0)
    s[S_SC1] = sc1 - dt * K_SC * sc1
    s[S_SC2] = sc2 + dt * (K_SC * sc1 - K_SC * sc2)


@njit(cache=True)
def initial_state(p):
    s = np.zeros(N_STATES)
    s[S_G] = p[P_GB]
    if p[P_IEXO] > 0.0 or p[P_S0] > 0.0:
        i_ss = p[P_S0] / p[P_N]
        x_ss = p[P_SI] * (i_ss + p[P_IEXO])
        s[S_I] = i_ss
        s[S_X] = x_ss
        s[S_G] = p[P_SG] * p[P_GB] / (p[P_SG] + x_ss)
    return s


@njit(cache=True)
def simulate_truth(params, carbs_fast, carbs_slow, ex, si_mult, egp_extra, hour, dt, burn_in):
    """Ground-truth simulation for a cohort.

    params: (P, N_PARAMS); inputs: (P, T) except hour: (T,). Returns plasma glucose (P, T)
    and plasma insulin (P, T). ``burn_in`` steps of the first day are run first to reach a
    realistic state.
    """
    P, T = carbs_fast.shape
    G_out = np.empty((P, T), dtype=np.float32)
    I_out = np.empty((P, T), dtype=np.float32)
    for i in range(P):
        p = params[i]
        s = initial_state(p)
        for t in range(burn_in):
            step(s, p, carbs_fast[i, t], carbs_slow[i, t], ex[i, t], si_mult[i, t],
                 egp_extra[i, t], hour[t], dt)
        for t in range(T):
            step(s, p, carbs_fast[i, t], carbs_slow[i, t], ex[i, t], si_mult[i, t],
                 egp_extra[i, t], hour[t], dt)
            G_out[i, t] = s[S_G]
            I_out[i, t] = s[S_I]
    return G_out, I_out


@njit(cache=True)
def observe(p, s0, cgm, carbs_fast, carbs_slow, ex, insulin, hour, dt):
    """Run the twin through observed data, assimilating CGM into the glucose state.

    Returns the post-assimilation state at every step, shape (T, N_STATES). With
    ``s0`` as the state *before* the first sample this works both in batch and
    incrementally (one sample at a time).
    """
    T = cgm.shape[0]
    s = s0.copy()
    out = np.empty((T, N_STATES))
    for t in range(T):
        s[S_SC1] += insulin[t]
        step(s, p, carbs_fast[t], carbs_slow[t], ex[t], 1.0, 0.0, hour[t], dt)
        if not math.isnan(cgm[t]):
            s[S_G] = cgm[t]
        out[t] = s
    return out


@njit(cache=True)
def forecast(p, states, hour, dt, n_ahead):
    """From every state, simulate n_ahead steps assuming no new meals or exercise.

    states: (T, N_STATES), hour: (T,). Returns (T, n_ahead) glucose forecasts.
    """
    T = states.shape[0]
    out = np.empty((T, n_ahead))
    for t in range(T):
        s = states[t].copy()
        h = hour[t]
        for k in range(n_ahead):
            h = h + dt / 60.0
            if h >= 24.0:
                h -= 24.0
            step(s, p, 0.0, 0.0, 0.0, 1.0, 0.0, h, dt)
            out[t, k] = s[S_G]
    return out


@njit(cache=True)
def forecast_with_inputs(p, states, carbs_fast, carbs_slow, ex, insulin, hour, dt, n_ahead, stride):
    """Retrospective forecasts that replay the *recorded* future inputs (calibration only).

    Starts from states[t] for t = 0, stride, 2·stride, ... and returns (len, n_ahead).
    """
    T = states.shape[0]
    n = (T + stride - 1) // stride
    out = np.full((n, n_ahead), np.nan)
    for j in range(n):
        t = j * stride
        s = states[t].copy()
        for k in range(n_ahead):
            u = t + k + 1
            if u >= T:
                break
            s[S_SC1] += insulin[u]
            step(s, p, carbs_fast[u], carbs_slow[u], ex[u], 1.0, 0.0, hour[u], dt)
            out[j, k] = s[S_G]
    return out


@njit(cache=True)
def rollout(p, s0, carbs_fast, carbs_slow, ex, si_mult, insulin, hour0, dt):
    """Open-loop scenario simulation from state s0 (used by the what-if simulator)."""
    T = carbs_fast.shape[0]
    s = s0.copy()
    out = np.empty(T)
    h = hour0
    for t in range(T):
        h = h + dt / 60.0
        if h >= 24.0:
            h -= 24.0
        s[S_SC1] += insulin[t]
        step(s, p, carbs_fast[t], carbs_slow[t], ex[t], si_mult[t], 0.0, h, dt)
        out[t] = s[S_G]
    return out
