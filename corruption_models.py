import numpy as np
from scipy.stats import ortho_group

DEFAULT_LOW_SPECTRUM_CONDITION_NUMBER = 1000


class Params:
    def __init__(self, d=20, n=10000, eps=0.05):
        self.d = d
        self.n = n
        self.eps = eps

def generate_mean(d, zero_mean=False, mean_scale=100.0):
    if zero_mean:
        return np.zeros(d)
    return np.random.uniform(-mean_scale, mean_scale, d)

def _ill_conditioned_covariance(d, kappa):
    U = ortho_group.rvs(d)
    eigenvalues = np.logspace(0, -np.log10(kappa), d)
    return U @ np.diag(eigenvalues) @ U.T


class CrossPolytopeModel:
    def __init__(self, zero_mean=False, mean_scale=10.0, radius_multiplier=5.0):
        self.zero_mean = zero_mean
        self.mean_scale = mean_scale
        self.radius_multiplier = radius_multiplier

    def generate(self, params):
        d, n, eps = params.d, params.n, params.eps
        U = ortho_group.rvs(d)[:, :1]
        Sigma_true = np.eye(d) + U @ np.diag([10.0]) @ U.T
        mu = generate_mean(d, zero_mean=self.zero_mean, mean_scale=self.mean_scale)
        X_clean = np.random.multivariate_normal(mu, Sigma_true, n)
        X_corrupted = X_clean.copy()

        num_corrupted = int(n * eps)
        R = self.radius_multiplier * d * np.log(d)
        outliers = np.zeros((num_corrupted, d))
        axes = np.random.randint(0, d, size=num_corrupted)
        signs = np.random.choice([-1, 1], size=num_corrupted)
        outliers[np.arange(num_corrupted), axes] = signs * R
        X_corrupted[:num_corrupted] = outliers + mu
        return X_clean, X_corrupted, Sigma_true

class InverseSpectrumModel:
    def __init__(self, variance_multiplier=1.0, zero_mean=False, mean_scale=100.0,
                 condition_number=DEFAULT_LOW_SPECTRUM_CONDITION_NUMBER):
        self.variance_multiplier = variance_multiplier
        self.zero_mean = zero_mean
        self.mean_scale = mean_scale
        self.condition_number = condition_number

    def generate(self, params):
        d, n, eps = params.d, params.n, params.eps
        Sigma_true = _ill_conditioned_covariance(d, self.condition_number)
        eigenvalues, U = np.linalg.eigh(Sigma_true)
        mu = generate_mean(d, zero_mean=self.zero_mean, mean_scale=self.mean_scale)
        X_clean = np.random.multivariate_normal(mu, Sigma_true, n)
        X_corrupted = X_clean.copy()

        num_corrupted = int(n * eps)
        v_min = U[:, 0]
        outlier_variance = eigenvalues[-1] * self.variance_multiplier
        outliers = np.outer(np.random.normal(0, np.sqrt(outlier_variance), num_corrupted), v_min) + mu
        X_corrupted[:num_corrupted] = outliers
        return X_clean, X_corrupted, Sigma_true

class ReversedSpectrumModel:
    def __init__(self, condition_number=DEFAULT_LOW_SPECTRUM_CONDITION_NUMBER,
                 offset=0.01, outlier_covariance_scale=0.3, zero_mean=False, mean_scale=100.0):
        self.condition_number = condition_number
        self.offset = offset
        self.outlier_covariance_scale = outlier_covariance_scale
        self.zero_mean = zero_mean
        self.mean_scale = mean_scale

    def generate(self, params):
        d, n, eps = params.d, params.n, params.eps
        Sigma_true = _ill_conditioned_covariance(d, self.condition_number)
        lambda_max = np.linalg.eigvalsh(Sigma_true)[-1]
        Sigma_out = self.outlier_covariance_scale * ((lambda_max + self.offset) * np.eye(d) - Sigma_true)
        Sigma_out = (Sigma_out + Sigma_out.T) / 2.0

        mu = generate_mean(d, zero_mean=self.zero_mean, mean_scale=self.mean_scale)
        X_clean = np.random.multivariate_normal(mu, Sigma_true, n)
        X_corrupted = X_clean.copy()
        num_corrupted = int(n * eps)
        outliers = np.random.multivariate_normal(mu, Sigma_out, num_corrupted)
        X_corrupted[:num_corrupted] = outliers
        return X_clean, X_corrupted, Sigma_true
