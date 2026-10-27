import numpy as np
import scipy.linalg as sla
import torch

PGD_TRIMMING_MULTIPLIER = 1.01
QUE_LIFTED_COVARIANCE_NORM_THRESHOLD = 2.4


class IterativeRefinement:
    def __init__(self, d, eps, rme_subroutine="PGD", c_step=0.05, pairwise=True):
        self.d = d
        self.pairwise = pairwise
        self.eps = 2.0 * eps if self.pairwise else eps
        self.rme_subroutine = rme_subroutine.upper()
        self.c_step = c_step
        self.idx_i, self.idx_j = np.triu_indices(self.d)
        self.vec_scale = np.where(self.idx_i == self.idx_j, 1.0, np.sqrt(2.0))

    def reduce_to_zero_mean(self, X):
        X = X[np.random.permutation(len(X)), :]
        half_n = len(X) // 2
        return (X[:half_n, :] - X[half_n:2 * half_n, :]) / np.sqrt(2.0)

    def project_capped_simplex(self, w, cap):
        tL, tR = np.min(w) - 1.0, np.max(w)
        for _ in range(50):
            t = (tL + tR) / 2.0
            if np.sum(np.clip(w - t, 0.0, cap)) < 1.0:
                tR = t
            else:
                tL = t
        return np.clip(w - tL, 0.0, cap)

    def _vec_to_sym_mat(self, v):
        M = np.zeros((self.d, self.d), dtype=v.dtype)
        M[self.idx_i, self.idx_j] = v / self.vec_scale
        M[self.idx_j, self.idx_i] = M[self.idx_i, self.idx_j]
        return M

    def _outer_products_to_vec(self, Y):
        # svec(Y_i Y_i^T), with sqrt(2) on off-diagonal entries.
        return Y[:, self.idx_i] * Y[:, self.idx_j] * self.vec_scale

    def _robust_mean_pgd(self, X, nItr=50, tol=1e-10, max_backtracking=20):
        n = X.shape[0]
        # Leave some room beyond the exact corruption fraction.
        epsn = int(round(PGD_TRIMMING_MULTIPLIER * self.eps * n))
        if epsn >= n:
            raise ValueError(f"Effective corruption budget too large: eps={self.eps:.4f}, n={n}.")
        cap = 1.0 / (n - epsn)

        w = np.ones(n) / n
        stepSz = 1.0 / n

        def top_eigenpair(weights):
            Xw = X.T @ weights
            covariance = X.T @ (weights[:, None] * X) - np.outer(Xw, Xw)
            values, vectors = sla.eigh(covariance, check_finite=False)
            return values[-1], Xw, vectors[:, -1]

        lambda_val, Xw, direction = top_eigenpair(w)
        for _ in range(nItr):
            Xu = X @ direction
            nabla_f_w = (Xu ** 2) - 2.0 * np.dot(Xw, direction) * Xu

            grad_norm = np.linalg.norm(nabla_f_w)
            if grad_norm < tol:
                break

            accepted = False
            for _ in range(max_backtracking):
                candidate_w = w - stepSz * nabla_f_w / grad_norm
                candidate_w = self.project_capped_simplex(candidate_w, cap)

                if np.linalg.norm(candidate_w - w, ord=1) < tol:
                    break

                new_lambda, new_Xw, new_direction = top_eigenpair(candidate_w)
                if new_lambda < lambda_val:
                    improvement = lambda_val - new_lambda
                    w, lambda_val, Xw, direction = candidate_w, new_lambda, new_Xw, new_direction
                    stepSz *= 2.0
                    accepted = True
                    break

                stepSz /= 4.0
                if stepSz < 1e-20:
                    break

            if not accepted or improvement < tol:
                break
        return X.T @ w

    def _robust_mean_que(self, Z_np, norm_thresh=QUE_LIFTED_COVARIANCE_NORM_THRESHOLD, n_iter=50):
        # The clean lifted covariance bound is 2, hence the 2.4 threshold.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z = torch.from_numpy(Z_np).to(torch.float64).to(device)

        n = Z.size(0)
        total_remove_count = int(self.eps * n)
        if total_remove_count >= n:
            raise ValueError(f"Effective corruption budget too large: eps={self.eps:.4f}, n={n}.")
        remove_per_iter = max(1, int(np.ceil(total_remove_count / n_iter)))
        min_n = n - total_remove_count

        def covariance_state(data):
            centered = data - data.mean(dim=0, keepdim=True)
            cov = torch.mm(centered.t(), centered) / data.size(0)
            vals, vectors = torch.linalg.eigh(cov)
            return centered, cov, vals[-1].item(), vectors[:, -1]

        Z_centered, covariance, spectral_norm, top_vector = covariance_state(Z)
        initial_spectral_norm = spectral_norm
        covariance_sum = torch.zeros_like(covariance)

        for _ in range(n_iter):
            if spectral_norm <= norm_thresh or Z.size(0) <= min_n:
                break

            # Keep past directions in the exponential score.
            covariance_sum += covariance
            evals, U = torch.linalg.eigh(covariance_sum / initial_spectral_norm)
            exp_evals = torch.exp(evals - evals.max())
            history_M = (U * exp_evals) @ U.t()
            history_M /= history_M.trace()

            # Also include the current top direction. Scores use float32.
            current_M = torch.outer(top_vector, top_vector)
            M = history_M + current_M
            M = (M / M.trace()).to(torch.float32)

            Z_centered_f32 = Z_centered.to(torch.float32)
            scores = (Z_centered_f32 * torch.mm(Z_centered_f32, M)).sum(dim=-1)

            n_keep = max(min_n, Z.size(0) - remove_per_iter)
            if Z.size(0) <= Z.size(1):
                break

            _, keep_idx = torch.topk(scores, k=n_keep, largest=False)
            Z = Z[keep_idx]
            Z_centered, covariance, spectral_norm, top_vector = covariance_state(Z)

        return Z.mean(dim=0).cpu().numpy()

    def _robust_second_moment(self, Y):
        Z = self._outer_products_to_vec(Y)

        if self.rme_subroutine == "PGD":
            M_vec = self._robust_mean_pgd(Z)
        elif self.rme_subroutine == "QUE":
            M_vec = self._robust_mean_que(Z)
        else:
            raise ValueError(f"Unknown rme_subroutine: {self.rme_subroutine}")

        return self._vec_to_sym_mat(M_vec)

    def robustCovEstimator(self, X_raw, max_outer_itr=10, outer_tol=1e-4,
                           *, initial_covariance=None):
        """Use an optional initial upper matrix in the original data units."""
        X = self.reduce_to_zero_mean(X_raw) if self.pairwise else X_raw
        sample_norms = np.linalg.norm(X, axis=1)
        scale = np.median(sample_norms)
        X = X / scale
        n = len(X)

        n_remove = int(round(self.eps * n))
        n_keep = n - n_remove
        if n_keep <= 0:
            raise ValueError(f"Effective corruption budget too large: eps={self.eps:.4f}, n={n}.")

        if initial_covariance is None:
            # CDGW19 initialization from the smallest-norm samples.
            keep_idx = np.argpartition(sample_norms, n_keep - 1)[:n_keep]
            X_init = X[keep_idx]
            Sigma_upper = 2.0 * ((X_init.T @ X_init) / n)
        else:
            Sigma_upper = np.asarray(initial_covariance) / (scale ** 2)
        slack_factor = self.c_step * np.sqrt(self.eps)

        prev_Sigma_hat = Sigma_hat = None
        for _ in range(max_outer_itr):
            evals, evecs = sla.eigh(Sigma_upper)
            evals = np.maximum(evals, 1e-12)
            Sigma_upper_inv_half = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            Sigma_upper_half = evecs @ np.diag(np.sqrt(evals)) @ evecs.T

            Y = X @ Sigma_upper_inv_half
            M = self._robust_second_moment(Y)

            Sigma_hat = Sigma_upper_half @ M @ Sigma_upper_half
            Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0
            Sigma_upper_next = Sigma_hat + slack_factor * Sigma_upper

            if prev_Sigma_hat is None:
                relative_diff = np.inf
            else:
                relative_diff = (
                    np.linalg.norm(Sigma_hat - prev_Sigma_hat)
                    / (np.linalg.norm(prev_Sigma_hat) + 1e-12)
                )
            if relative_diff < outer_tol:
                break
            prev_Sigma_hat = Sigma_hat
            Sigma_upper = Sigma_upper_next

        return Sigma_hat * (scale ** 2)
