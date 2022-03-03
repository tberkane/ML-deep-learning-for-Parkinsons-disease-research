# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_data(path_dataset, sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=2, missing_values = "", filling_values = -999)
    return data


def sample_data(y, x, seed):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y, x


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
