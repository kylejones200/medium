"""Core functions for survival analysis time-to-failure modeling."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from lifelines import KaplanMeierFitter, WeibullFitter, CoxPHFitter
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_survival_data(data_path: Path) -> pd.DataFrame:
    """Load survival analysis dataset."""
    return pd.read_csv(data_path)

def fit_kaplan_meier(df: pd.DataFrame, duration_col: str, event_col: str) -> KaplanMeierFitter:
    """Fit Kaplan-Meier survival estimator."""
    kmf = KaplanMeierFitter()
    kmf.fit(df[duration_col], df[event_col])
    return kmf

def fit_weibull_survival(df: pd.DataFrame, duration_col: str, event_col: str) -> WeibullFitter:
    """Fit Weibull survival model."""
    wf = WeibullFitter()
    wf.fit(df[duration_col], df[event_col])
    return wf

def fit_cox_proportional_hazards(df: pd.DataFrame, duration_col: str, event_col: str,
                                covariates: list) -> CoxPHFitter:
    """Fit Cox Proportional Hazards model."""
    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
    return cph

def plot_survival_curve(kmf: KaplanMeierFitter, title: str, output_path: Path):
 """Plot survival curve """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

