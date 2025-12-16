"""Core functions for causal inference in time series econometrics."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def perform_granger_causality_test(data: pd.DataFrame, var1: str, var2: str,
                                   maxlag: int = 4) -> dict:
    """Perform Granger causality test."""
    test_data = data[[var1, var2]]
    result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
    return result

def difference_in_differences(y_treated: np.ndarray, y_control: np.ndarray,
                              treatment_period: int) -> float:
    """Calculate difference-in-differences estimator."""
    pre_treated = y_treated[:treatment_period].mean()
    post_treated = y_treated[treatment_period:].mean()
    pre_control = y_control[:treatment_period].mean()
    post_control = y_control[treatment_period:].mean()
    
    did = (post_treated - pre_treated) - (post_control - pre_control)
    return did

def plot_causal_effect(y_treated: np.ndarray, y_control: np.ndarray,
                      treatment_period: int, output_path: Path):
 """Plot causal effect """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = np.arange(len(y_treated))
    ax.plot(time, y_treated, label="Treated", color="#4A90A4", linewidth=1.2)
    ax.plot(time, y_control, label="Control", color="#D4A574", linewidth=1.2)
    ax.axvline(treatment_period, color='red', linestyle='--', linewidth=1.2, label='Treatment')
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

