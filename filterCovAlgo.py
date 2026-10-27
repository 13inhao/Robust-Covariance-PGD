import numpy as np


class FilterCovariance:
    def __init__(self, d, eps, expConst=2.0, tau=0.1,
                 max_bracket_trials=20, max_binary_trials=50,
                 target_lower_mult=0.95, target_upper_mult=1.05, pairwise=False):
        self.d = d
        self.pairwise = pairwise
        self.eps = 2.0 * eps if self.pairwise else eps

        self.expConst = float(expConst)
        self.tau = float(tau)
        self.max_bracket_trials = max_bracket_trials
        self.max_binary_trials = max_binary_trials
        self.target_lower_mult = float(target_lower_mult)
        self.target_upper_mult = float(target_upper_mult)

        if not 0.0 <= self.target_lower_mult <= self.target_upper_mult:
            raise ValueError("Require 0 <= target_lower_mult <= target_upper_mult.")

        self.threshold = self.eps * (np.log(1.0 / self.eps)) ** 2
        self.C1 = 0.4
        self.C2 = 0.0

    def reduce_to_zero_mean(self, X):
        X = X[np.random.permutation(len(X)), :]
        half_n = len(X) // 2
        return (X[:half_n, :] - X[half_n:2 * half_n, :]) / np.sqrt(2.0)

    def _inv_sqrt_spd(self, A, eps_reg=1e-12):
        evals, evecs = np.linalg.eigh(A)
        evals = np.maximum(evals, eps_reg)
        inv_sqrt_evals = 1.0 / np.sqrt(evals)
        return (evecs * inv_sqrt_evals) @ evecs.T

    def _find_max_poly(self, data):
        n, d = data.shape
        v = np.eye(d).reshape(-1, order="C")
        Y = np.einsum("ni,nj->nij", data, data).reshape(n, d * d, order="C")
        Sigma_Y = (Y.T @ Y) / n - np.outer(v, v)
        evals, evecs = np.linalg.eigh(Sigma_Y)
        idx = np.argmax(evals)
        lambda_val = evals[idx] / 2.0
        M = evecs[:, idx].reshape((d, d), order="C") / np.sqrt(2.0)
        return M, lambda_val

    def _filter_gaussian_cov(self, data, expConst=None):
        if expConst is None:
            expConst = self.expConst

        n, d = data.shape

        if n == 0:
            return np.zeros((d, d)), 0

        Sigma_emp = (data.T @ data) / n
        condition = np.linalg.cond(Sigma_emp)

        if condition > d**8 or n < d:
            return np.zeros((d, d)), 0

        Sigma_emp_inv = np.linalg.inv(Sigma_emp)

        quad = np.einsum("ij,jk,ik->i", data, Sigma_emp_inv, data)
        cutoff = self.C1 * d * np.log(n / self.tau)
        remove_mask = quad > cutoff

        if np.any(remove_mask):
            return self._filter_gaussian_cov(data[~remove_mask], expConst=expConst)

        Sigma_emp_inv_sqrt = self._inv_sqrt_spd(Sigma_emp)
        X_rot = data @ Sigma_emp_inv_sqrt

        M, lambda_val = self._find_max_poly(X_rot)

        if lambda_val < 1 + self.C2 * self.threshold:
            return Sigma_emp, n

        X_proj = np.einsum("ij,jk,ik->i", X_rot, M, X_rot)
        indices = np.abs(X_proj - np.mean(X_proj))

        sort_idx = np.argsort(indices)
        sorted_indices = indices[sort_idx]

        log_term = np.log(n / self.tau)
        noise_term = 3.0 * self.eps / (d * log_term) ** 2

        break_idx = None
        for i in range(n):
            T = sorted_indices[i]
            rhs = n * (12.0 * np.exp(-expConst * T) + noise_term)
            rem_points = n - i - 1

            if rem_points > rhs:
                break_idx = i
                break

        if break_idx is None or break_idx == 0:
            return Sigma_emp, n

        return self._filter_gaussian_cov(data[sort_idx[:break_idx + 1]], expConst=expConst)

    def _filter_gaussian_cov_tuned(self, data):
        expConstUB = np.inf
        expConstLB = 0.0
        expConstCand = float(self.expConst)

        n = data.shape[0]
        target_noise = self.eps * n
        target_retained = n - target_noise
        lower_target = n - self.target_upper_mult * target_noise
        upper_target = n - self.target_lower_mult * target_noise

        best_result = None
        best_target_gap = np.inf

        def evaluate_candidate(candidate):
            nonlocal best_result, best_target_gap
            covariance, n_out = self._filter_gaussian_cov(data, expConst=candidate)
            target_gap = abs(n_out - target_retained)
            if target_gap < best_target_gap:
                best_target_gap = target_gap
                best_result = covariance
            return covariance, n_out

        Sigma_hat, n_out = evaluate_candidate(expConstCand)
        bracket_trials = 0

        if n_out < lower_target:
            while n_out < lower_target and bracket_trials < self.max_bracket_trials:
                bracket_trials += 1
                expConstUB = expConstCand
                expConstCand /= 2.0

                Sigma_hat, n_out = evaluate_candidate(expConstCand)

            expConstLB = expConstCand

        elif n_out > upper_target:
            while n_out > upper_target and bracket_trials < self.max_bracket_trials:
                bracket_trials += 1
                expConstLB = expConstCand
                expConstCand *= 2.0

                Sigma_hat, n_out = evaluate_candidate(expConstCand)

            expConstUB = expConstCand

        if lower_target < n_out < upper_target:
            return Sigma_hat

        if not np.isfinite(expConstUB):
            return best_result

        for _ in range(self.max_binary_trials):
            expConstCand = 0.5 * (expConstUB + expConstLB)

            Sigma_hat, n_out = evaluate_candidate(expConstCand)

            if n_out < lower_target:
                expConstUB = expConstCand
            elif n_out > upper_target:
                expConstLB = expConstCand
            else:
                return Sigma_hat

        # If removals jump over the target interval, use the closest count.
        return best_result

    def robustCovEstimator(self, X_raw):
        X = self.reduce_to_zero_mean(X_raw) if self.pairwise else X_raw
        return self._filter_gaussian_cov_tuned(X)
