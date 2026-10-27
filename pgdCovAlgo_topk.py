import numpy as np
import scipy.linalg as sla


class RobustCovariancePGD_TopK:
    """Genomic PGD over a sphere grid in the current top-three eigenspace."""

    def __init__(self, d, eps, pairwise=True, n_sphere=4000):
        self.d = d
        self.pairwise = pairwise
        self.eps = 2.0 * eps if self.pairwise else eps
        self.n_sphere = n_sphere
        self._sphere3 = self._fib_sphere(n_sphere)

    def _fib_sphere(self, N):
        """Fibonacci grid of unit directions in three dimensions."""
        i = np.arange(N)
        gold = (1.0 + 5.0 ** 0.5) / 2.0
        z = 1.0 - 2.0 * (i + 0.5) / N
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
        theta = 2.0 * np.pi * i / gold
        return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

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

    def _dir_terms(self, proj, w, coeff):
        """Objective and weight gradient for a fixed direction."""
        a = proj * proj
        aw = a @ w
        sum_w_a2 = (a * a) @ w
        A = sum_w_a2 - aw * aw
        B = aw * aw
        obj = A - coeff * B
        grad = a * a - 2.0 * (1.0 + coeff) * aw * a
        return obj, grad

    def _inner(self, X, w, coeff):
        Sigma_w = X.T @ (w[:, None] * X)
        _, evecs = sla.eigh(Sigma_w)
        E3 = evecs[:, ::-1][:, :3]
        P3 = X @ E3
        wP = w[:, None] * P3
        S2 = P3.T @ wP

        # Evaluate the grid from 3D moments, without a grid-by-sample array.
        T = np.einsum('ij,ik,il,im->jklm', wP, P3, P3, P3)
        Tmat = T.reshape(9, 9)
        dirs = self._sphere3
        aw_all = np.einsum('gj,jk,gk->g', dirs, S2, dirs)
        dd = (dirs[:, :, None] * dirs[:, None, :]).reshape(dirs.shape[0], 9)
        sum_w_a2_all = np.einsum('gp,pq,gq->g', dd, Tmat, dd)

        obj_all = sum_w_a2_all - (1.0 + coeff) * aw_all * aw_all
        t = int(np.argmax(obj_all))
        v_star = E3 @ dirs[t]
        obj, grad = self._dir_terms(X @ v_star, w, coeff)

        # Select across coefficients by the ratio, not the PGD objective.
        ratio_all = sum_w_a2_all / (aw_all * aw_all)
        score = float(np.max(ratio_all))
        return obj, grad, score

    def _run_pgd_fixed_coeff(self, X, n, cap, coeff, max_itr):
        w = np.ones(n) / n
        step_size = 1.0 / n
        for _ in range(max_itr):
            old_obj, grad, _ = self._inner(X, w, coeff)
            old_w = w
            w = self.project_capped_simplex(w - step_size * grad, cap)

            new_obj, _, _ = self._inner(X, w, coeff)
            if new_obj < old_obj:
                step_size = min(step_size * 2.0, 1.0)
            else:
                step_size /= 4.0
                w = old_w
                if step_size < 1e-20:
                    break

        # Score the returned weights, including the last accepted step.
        _, _, score = self._inner(X, w, coeff)
        return w, score

    def robustCovEstimator(self, X_raw, max_itr=300, coeffs=None):
        X = self.reduce_to_zero_mean(X_raw) if self.pairwise else X_raw
        scale = np.median(np.linalg.norm(X, axis=1))
        X = X / scale
        n = len(X)

        epsn = int(round(self.eps * n))
        if epsn >= n:
            raise ValueError(f"Effective corruption budget too large: eps={self.eps:.4f}, n={n}.")
        cap = 1.0 / (n - epsn)

        if coeffs is None:
            coeffs = list(np.arange(1.0, 5.2, 0.2)) + list(range(6, 21))

        best_w, best_score, best_coeff = None, np.inf, None
        for coeff in coeffs:
            w, score = self._run_pgd_fixed_coeff(X, n, cap, coeff, max_itr)
            if score < best_score:
                best_score, best_w, best_coeff = score, w.copy(), coeff

        self.coeff = best_coeff
        Sigma_robust_scaled = X.T @ (best_w[:, None] * X)
        return Sigma_robust_scaled * (scale ** 2)


def error_Mahalanobis(Sigma_hat, Sigma_true):
    ev, evec = sla.eigh(Sigma_true)
    inv_half = (evec / np.sqrt(ev)) @ evec.T
    M = inv_half @ Sigma_hat @ inv_half
    return np.linalg.norm(M - np.eye(Sigma_true.shape[0]), 'fro')


def error_Mahalanobis_spectral(Sigma_hat, Sigma_true):
    ev, evec = sla.eigh(Sigma_true)
    inv_half = (evec / np.sqrt(ev)) @ evec.T
    M = inv_half @ Sigma_hat @ inv_half
    return np.linalg.norm(M - np.eye(Sigma_true.shape[0]), 2)


def top2_subspace_error(Sigma_hat, Sigma_true):
    """Frobenius distance between the top-two orthogonal projectors."""
    U = sla.eigh(Sigma_hat)[1][:, ::-1][:, :2]
    V = sla.eigh(Sigma_true)[1][:, ::-1][:, :2]
    return np.linalg.norm(U @ U.T - V @ V.T, 'fro')
