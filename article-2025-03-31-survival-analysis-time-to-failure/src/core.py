"""Core functions for survival analysis time-to-failure modeling."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from lifelines import KaplanMeierFitter, WeibullFitter, CoxPHFitter
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


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
    """Plot survival curve with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    kmf.plot_survival_function(ax=ax, color="#4A90A4", linewidth=1.2)
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    
    save_tufte_figure(output_path)

