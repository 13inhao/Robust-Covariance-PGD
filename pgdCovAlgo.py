import numpy as np
import scipy.linalg as sla


class RobustCovariancePGD:
    """Robust covariance estimation by capped-simplex projected GD."""

    def __init__(self, d, eps, pairwise=True, coeff_start=5.0,
                 coeff_min=2.05, coeff_step=0.05, continuation_max_itr=50,
                 trimming_multiplier=1.01):
        self.d = d
        self.pairwise = pairwise
        self.eps = 2.0 * eps if self.pairwise else eps
        self.coeff_start = float(coeff_start)
        self.coeff_min = float(coeff_min)
        self.coeff_step = float(coeff_step)
        self.continuation_max_itr = int(continuation_max_itr)
        self.trimming_multiplier = float(trimming_multiplier)

        self.idx_i, self.idx_j = np.triu_indices(self.d)
        self.vec_scale = np.where(self.idx_i == self.idx_j, 1.0, np.sqrt(2.0))
        self.kron_basis_scale = np.where(
            self.idx_i == self.idx_j, 0.5, 1.0 / np.sqrt(2.0)
        )

    def _vec_to_sym_mat(self, v):
        M = np.zeros((self.d, self.d), dtype=v.dtype)
        M[self.idx_i, self.idx_j] = v / self.vec_scale
        M[self.idx_j, self.idx_i] = M[self.idx_i, self.idx_j]
        return M

    def _reduce_to_zero_mean(self, X):
        X = X[np.random.permutation(len(X))]
        half_n = len(X) // 2
        return (X[:half_n] - X[half_n:2 * half_n]) / np.sqrt(2.0)

    def _project_capped_simplex(self, w, cap):
        lower, upper = np.min(w) - 1.0, np.max(w)
        for _ in range(50):
            threshold = (lower + upper) / 2.0
            if np.sum(np.clip(w - threshold, 0.0, cap)) < 1.0:
                upper = threshold
            else:
                lower = threshold
        return np.clip(w - lower, 0.0, cap)

    def _compute_Q_kron(self, S):
        """kron(S, S) restricted to the symmetric-vector basis."""
        i = self.idx_i[:, None]
        j = self.idx_j[:, None]
        k = self.idx_i[None, :]
        l = self.idx_j[None, :]

        Q_kron = S[j, k] * S[l, i]
        Q_kron += S[j, l] * S[k, i]
        Q_kron += S[i, k] * S[l, j]
        Q_kron += S[i, l] * S[k, j]
        Q_kron *= self.kron_basis_scale[:, None]
        Q_kron *= self.kron_basis_scale[None, :]
        return Q_kron

    def _compute_Q_Y(self, w, Z):
        mu_w_Z = Z.T @ w
        return Z.T @ (w[:, None] * Z) - np.outer(mu_w_Z, mu_w_Z)

    def _direction_gradient(self, X, Z, w, Sigma_w_X, direction, coeff):
        """Gradient with the top eigendirection held fixed."""
        scores = Z @ direction
        U = self._vec_to_sym_mat(direction)
        U_S_U = U @ Sigma_w_X @ U
        return (
            scores**2
            - 2.0 * (scores @ w) * scores
            - 2.0 * coeff * np.sum(X * (X @ U_S_U), axis=1)
        )

    def _run_pgd_fixed_coeff(self, X, Z, cap, coeff, max_itr, w, step_size):
        w = w.copy()

        Sigma_w_X = X.T @ (w[:, None] * X)
        Q_Y = self._compute_Q_Y(w, Z)
        Q_kron = self._compute_Q_kron(Sigma_w_X)

        need_grad = True
        for n_itr in range(1, max_itr + 1):
            if need_grad:
                Q = Q_Y - coeff * Q_kron
                evals, evecs = sla.eigh(Q)
                old_f_w = evals[-1]
                if old_f_w < 0.0:
                    break

                gradient = self._direction_gradient(
                    X, Z, w, Sigma_w_X, evecs[:, -1], coeff
                )
                need_grad = False

            old_w = w
            w = self._project_capped_simplex(w - step_size * gradient, cap)

            if np.array_equal(w, old_w):
                break

            Sigma_w_X_new = X.T @ (w[:, None] * X)
            Q_Y_new = self._compute_Q_Y(w, Z)
            Q_kron_new = self._compute_Q_kron(Sigma_w_X_new)
            f_w = sla.eigvalsh(Q_Y_new - coeff * Q_kron_new)[-1]

            if f_w < old_f_w:
                step_size = min(step_size * 2.0, 1.0)
                Sigma_w_X = Sigma_w_X_new
                Q_Y = Q_Y_new
                Q_kron = Q_kron_new
                need_grad = True
            else:
                step_size /= 4.0
                w = old_w
                if step_size < 1e-20:
                    break

        final_f_w = sla.eigvalsh(Q_Y - coeff * Q_kron)[-1]
        return w, Q_Y, Q_kron, final_f_w, n_itr, step_size

    def _run_pgd(self, X, Z, cap, max_itr):
        """Lower coeff with warm starts and keep the last certified solution."""
        n = X.shape[0]
        coeff = self.coeff_start
        w = np.ones(n) / n
        step_size = 1.0 / n
        last_success = None
        total_itr = 0

        while total_itr < max_itr and coeff >= self.coeff_min - 1e-12:
            remaining_itr = max_itr - total_itr
            stage_max_itr = (
                remaining_itr
                if last_success is None
                else min(remaining_itr, self.continuation_max_itr)
            )
            w, Q_Y, Q_kron, final_f_w, n_itr, step_size = self._run_pgd_fixed_coeff(
                X, Z, cap, coeff, stage_max_itr, w, step_size
            )
            total_itr += n_itr

            if final_f_w < 0.0:
                last_success = (w.copy(), coeff)

            if final_f_w >= 0.0:
                break

            step_size = max(step_size, 1.0 / n)
            next_coeff = round(coeff - self.coeff_step, 10)

            # Reuse Q while the weights stay fixed.
            while total_itr < max_itr and next_coeff >= self.coeff_min - 1e-12:
                next_f_w = sla.eigvalsh(Q_Y - next_coeff * Q_kron)[-1]
                if next_f_w >= 0.0:
                    break

                coeff = next_coeff
                total_itr += 1
                last_success = (w.copy(), coeff)
                next_coeff = round(coeff - self.coeff_step, 10)

            coeff = next_coeff

        return last_success

    def robustCovEstimator(self, X_raw, max_itr=1000):
        X_raw = np.asarray(X_raw)
        X = self._reduce_to_zero_mean(X_raw) if self.pairwise else X_raw

        scale = np.median(np.linalg.norm(X, axis=1))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("The median sample norm must be positive and finite.")
        X = X / scale
        n = X.shape[0]

        # Form the symmetric lift without storing full outer products.
        Z = X[:, self.idx_i] * X[:, self.idx_j] * self.vec_scale

        trimming_eps = self.trimming_multiplier * self.eps
        epsn = int(round(trimming_eps * n))
        if epsn >= n:
            raise ValueError(
                f"Effective corruption budget is too large: "
                f"eps={trimming_eps:.4f}, n={n}, epsn={epsn}."
            )
        cap = 1.0 / (n - epsn)

        result = self._run_pgd(X, Z, cap, max_itr)
        if result is None:
            raise RuntimeError(
                "PGD failed to find a certified negative-objective solution."
            )

        best_w, best_coeff = result
        Sigma_robust_scaled = X.T @ (best_w[:, None] * X)

        self.coeff = best_coeff

        return Sigma_robust_scaled * (scale ** 2)


def error_Mahalanobis(Sigma_hat, Sigma_true):
    """Return the Frobenius multiplicative covariance error."""
    eigvals, eigvecs = sla.eigh(Sigma_true)
    Sigma_true_inv_half = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
    M = Sigma_true_inv_half @ Sigma_hat @ Sigma_true_inv_half
    return np.linalg.norm(M - np.eye(Sigma_true.shape[0]), ord='fro')


def error_Mahalanobis_spectral(Sigma_hat, Sigma_true):
    """Return the spectral-norm multiplicative covariance error."""
    eigvals, eigvecs = sla.eigh(Sigma_true)
    Sigma_true_inv_half = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
    M = Sigma_true_inv_half @ Sigma_hat @ Sigma_true_inv_half
    return np.linalg.norm(M - np.eye(Sigma_true.shape[0]), ord=2)
