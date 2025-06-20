# Creating the full Python script with all cross-validation classes and utilities from the article

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import acf


class TimeSeriesCV:
    def __init__(self, data, date_column, target_column):
        self.data = data
        self.date_column = date_column
        self.target_column = target_column

    def plot_time_series_split(self, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fig, axs = plt.subplots(n_splits, 1, figsize=(15, 5 * n_splits))
        for idx, (train_idx, test_idx) in enumerate(tscv.split(self.data)):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            axs[idx].plot(train_data[self.date_column], train_data[self.target_column], label='Training')
            axs[idx].plot(test_data[self.date_column], test_data[self.target_column], label='Validation')
            axs[idx].set_title(f'Split {idx + 1}')
            axs[idx].legend()
        plt.tight_layout()
        return fig


class RollingWindowCV:
    def __init__(self, window_size, step_size=1):
        self.window_size = window_size
        self.step_size = step_size

    def split(self, data):
        n_samples = len(data)
        indices = np.arange(n_samples)
        for start_idx in range(0, n_samples - self.window_size, self.step_size):
            end_idx = start_idx + self.window_size
            if end_idx + self.step_size <= n_samples:
                train_idx = indices[start_idx:end_idx]
                test_idx = indices[end_idx:end_idx + self.step_size]
                yield train_idx, test_idx

    def evaluate_model(self, model, data, target):
        scores = []
        predictions = []
        for train_idx, test_idx in self.split(data):
            X_train = data.iloc[train_idx]
            X_test = data.iloc[test_idx]
            y_train = target.iloc[train_idx]
            y_test = target.iloc[test_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores.append(self._calculate_metrics(y_test, pred))
            predictions.extend(pred)
        return scores, predictions

    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }


class NestedTimeSeriesCV:
    def __init__(self, n_splits_outer=5, n_splits_inner=3):
        self.n_splits_outer = n_splits_outer
        self.n_splits_inner = n_splits_inner

    def run_nested_cv(self, model, param_grid, X, y):
        outer_cv = TimeSeriesSplit(n_splits=self.n_splits_outer)
        inner_cv = TimeSeriesSplit(n_splits=self.n_splits_inner)
        outer_scores = []
        best_params = []
        for outer_train_idx, outer_test_idx in outer_cv.split(X):
            X_train_outer = X.iloc[outer_train_idx]
            X_test_outer = X.iloc[outer_test_idx]
            y_train_outer = y.iloc[outer_train_idx]
            y_test_outer = y.iloc[outer_test_idx]
            best_score = float('inf')
            best_param = None
            for params in ParameterGrid(param_grid):
                inner_scores = []
                for inner_train_idx, inner_test_idx in inner_cv.split(X_train_outer):
                    X_train_inner = X_train_outer.iloc[inner_train_idx]
                    X_test_inner = X_train_outer.iloc[inner_test_idx]
                    y_train_inner = y_train_outer.iloc[inner_train_idx]
                    y_test_inner = y_train_outer.iloc[inner_test_idx]
                    model.set_params(**params)
                    model.fit(X_train_inner, y_train_inner)
                    pred = model.predict(X_test_inner)
                    score = mean_squared_error(y_test_inner, pred)
                    inner_scores.append(score)
                avg_score = np.mean(inner_scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_param = params
            model.set_params(**best_param)
            model.fit(X_train_outer, y_train_outer)
            pred = model.predict(X_test_outer)
            score = mean_squared_error(y_test_outer, pred)
            outer_scores.append(score)
            best_params.append(best_param)
        return outer_scores, best_params


class BlockingTimeSeriesCV:
    def __init__(self, block_size, n_splits=5):
        self.block_size = block_size
        self.n_splits = n_splits

    def split(self, data):
        n_samples = len(data)
        n_blocks = n_samples // self.block_size
        indices = np.arange(n_samples)
        blocks = np.array_split(indices[:n_blocks * self.block_size], n_blocks)
        for i in range(self.n_splits):
            test_block_idx = i % n_blocks
            test_indices = blocks[test_block_idx]
            train_indices = np.concatenate([
                block for j, block in enumerate(blocks) if j != test_block_idx
            ])
            yield train_indices, test_indices


class TimeSeriesEvaluation:
    def __init__(self):
        self.metrics = {}

    def add_metric(self, name, function):
        self.metrics[name] = function

    def evaluate(self, y_true, y_pred):
        results = {}
        for name, function in self.metrics.items():
            results[name] = function(y_true, y_pred)
        return results

    def cross_validate(self, model, cv_splitter, X, y):
        cv_results = {name: [] for name in self.metrics.keys()}
        for train_idx, test_idx in cv_splitter.split(X):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results = self.evaluate(y_test, y_pred)
            for name, value in results.items():
                cv_results[name].append(value)
        return {name: np.mean(scores) for name, scores in cv_results.items()}


def check_temporal_dependencies(data, max_lag=10):
    return acf(data, nlags=max_lag)


def check_data_leakage(train_indices, test_indices, timestamps):
    train_dates = timestamps[train_indices]
    test_dates = timestamps[test_indices]
    return np.all(train_dates < test_dates.min())
