import argparse
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from pgdCovAlgo import error_Mahalanobis, RobustCovariancePGD as PGD_Cov
from filterCovAlgo import FilterCovariance as Filtering
from iterRefAlgo import IterativeRefinement as IterRef
from corruption_models import Params, CrossPolytopeModel, InverseSpectrumModel, ReversedSpectrumModel

SEED = 1520761101
DEFAULT_DIMS = list(range(5, 31, 5))
EPS = 0.10
SAMPLE_SIZE_MULTIPLIER = 2.0
FIXED_D = 10
DEFAULT_EPS_LIST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N = 100000
N_REPEAT = 20
ZERO_MEAN = True
ADAPTIVE_COEFF_START = 5.0
ADAPTIVE_COEFF_MIN = 2.05
ADAPTIVE_COEFF_STEP = 0.05

ALGO_LIST = ["PGD Covariance", "Filtering", "IterRef PGD", "IterRef QUE"]
CORRUPTION_MODELS = [
    (CrossPolytopeModel, {}),
    (InverseSpectrumModel, {"variance_multiplier": 1.0}),
    (ReversedSpectrumModel, {}),
]
MODEL_LABELS = {
    "CrossPolytopeModel": "(a) Cross-Polytope",
    "InverseSpectrumModel": "(b) Inverse-Spectrum",
    "ReversedSpectrumModel": "(c) Reversed-Spectrum",
}
COMBINE_MODEL_PANELS = True
STYLES = {
    "PGD Covariance": ("r", "D", "-"),
    "Filtering": ("b", "s", "-."),
    "IterRef PGD": ("m", "^", ":"),
    "IterRef QUE": ("c", "o", "--"),
}


def sample_size_for_dim(d, eps):
    return int(round(SAMPLE_SIZE_MULTIPLIER * d * (d + 1) / eps**2))


def make_estimator(algo, params):
    if algo == "PGD Covariance":
        return PGD_Cov(params.d, params.eps, pairwise=not ZERO_MEAN,
                       coeff_start=ADAPTIVE_COEFF_START, coeff_min=ADAPTIVE_COEFF_MIN,
                       coeff_step=ADAPTIVE_COEFF_STEP)
    if algo == "Filtering":
        return Filtering(params.d, params.eps, pairwise=not ZERO_MEAN)
    if algo in ("IterRef PGD", "IterRef QUE"):
        return IterRef(params.d, params.eps, rme_subroutine=algo.split()[-1], pairwise=not ZERO_MEAN)
    raise ValueError(f"Unknown algorithm: {algo}")


def make_corruption_model(model_class, model_kwargs):
    return model_class(**{"zero_mean": ZERO_MEAN, **model_kwargs})


def run_trials(model, params, n_repeat, algorithms, make_estimator):
    datasets, sampling_errors, naive_errors = [], [], []
    for _ in range(n_repeat):
        X_clean, X_corrupted, Sigma_true = model.generate(params)
        Sigma_clean = (X_clean.T @ X_clean) / params.n
        sampling_errors.append(error_Mahalanobis(Sigma_clean, Sigma_true))
        X_centered = X_corrupted - np.mean(X_corrupted, axis=0)
        Sigma_naive = (X_centered.T @ X_centered) / params.n
        naive_errors.append(error_Mahalanobis(Sigma_naive, Sigma_true))
        datasets.append((X_corrupted, Sigma_true))

    results = {"sampling_error": sampling_errors, "naive_error": naive_errors}
    print(f"Sampling error: {np.mean(sampling_errors):.6f}, naive error: {np.mean(naive_errors):.6f}")
    for algo in algorithms:
        raw_errors, excess_errors, runtimes = [], [], []
        for trial, (X_corrupted, Sigma_true) in enumerate(datasets):
            estimator = make_estimator(algo, params)
            start = time.time()
            try:
                Sigma_hat = estimator.robustCovEstimator(X_corrupted)
                error = error_Mahalanobis(Sigma_hat, Sigma_true)
                raw_errors.append(error)
                excess_errors.append(max(0, error - sampling_errors[trial]))
            except Exception as e:
                print(f"{algo}, trial {trial + 1} failed: {e}")
                raw_errors.append(np.nan)
                excess_errors.append(np.nan)
            runtimes.append(time.time() - start)
        results[algo] = {"raw": raw_errors, "excess": excess_errors}
        print(f"{algo}: error={np.nanmean(raw_errors):.6f}, "
              f"excess={np.nanmean(excess_errors):.6f}, time={np.mean(runtimes):.2f}s")
    return results


def append_trials(results, trials, algorithms):
    for key in ("sampling_error", "naive_error"):
        results[key].append(trials[key])
    for algo in algorithms:
        for key in ("raw", "excess"):
            results[algo][key].append(trials[algo][key])


def mean_and_se(data, n_repeat):
    data = np.asarray(data, dtype=float)
    return np.nanmean(data, axis=1), np.nanstd(data, axis=1, ddof=1) / np.sqrt(n_repeat)


def raw_error_limits(result_items, n_repeat, algorithms):
    lower, upper = np.inf, -np.inf
    for _, results in result_items:
        curves = [results[algo]["raw"] for algo in algorithms] + [results["sampling_error"]]
        for data in curves:
            mean, se = mean_and_se(data, n_repeat)
            lower = min(lower, np.nanmin(mean - se))
            upper = max(upper, np.nanmax(mean + se))
    padding = 0.1 * max(upper - lower, 0.01)
    y_min = max(0.0, np.floor((lower - padding) / 0.01) * 0.01)
    y_max = np.ceil((upper + padding) / 0.01) * 0.01
    return y_min, y_max


def plot_sweep(result_items, xs, xlabel, n_repeat, styles, model_labels,
               combined_path, separate_path, combine=True):
    def draw_panel(ax, model_name, results):
        scale = 2.1 if combine else 1.0
        for algo, (color, marker, line) in styles.items():
            mean, se = mean_and_se(results[algo]["raw"], n_repeat)
            ax.plot(xs, mean, color=color, marker=marker, linestyle=line,
                    linewidth=2 * scale, markersize=6 * scale, label=algo)
            ax.fill_between(xs, mean - se, mean + se, color=color, alpha=0.10)

        mean, se = mean_and_se(results["sampling_error"], n_repeat)
        ax.plot(xs, mean, color="k", linestyle="--", linewidth=2 * scale, label="Sampling baseline")
        ax.fill_between(xs, mean - se, mean + se, color="gray", alpha=0.10)
        ax.set_title(model_labels[model_name], fontsize=14 * scale)
        ax.set_xlabel(xlabel, fontsize=12 * scale)
        ax.set_xticks(xs)
        ax.tick_params(axis="both", labelsize=10 * scale)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)

    if combine:
        fig, axes = plt.subplots(1, len(result_items), figsize=(6.3 * len(result_items), 6.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (model_name, results) in zip(axes, result_items):
            draw_panel(ax, model_name, results)
        axes[0].set_ylabel("Mahalanobis Error", fontsize=25)
        axes[0].set_ylim(*raw_error_limits(result_items, n_repeat, styles))
        axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.02), fontsize=20, frameon=False)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        fig.savefig(combined_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        for model_name, results in result_items:
            fig, ax = plt.subplots(figsize=(10, 6))
            draw_panel(ax, model_name, results)
            ax.set_ylabel("Mahalanobis Error", fontsize=12)
            ax.set_ylim(*raw_error_limits([(model_name, results)], n_repeat, styles))
            ax.legend()
            fig.tight_layout()
            fig.savefig(separate_path.format(model=model_name), format="pdf", bbox_inches="tight")
            plt.close(fig)


def run_sweep(sweep, values=None, plot_only=False):
    np.random.seed(SEED)
    if sweep == "dimension":
        values = DEFAULT_DIMS if values is None else values
        ns = [sample_size_for_dim(d, EPS) for d in values]
        settings = [(d, n, EPS) for d, n in zip(values, ns)]
        metadata = {
            "ds": values, "ns": ns, "n_repeat": N_REPEAT, "eps": EPS,
            "sample_size_rule": "int(round(SAMPLE_SIZE_MULTIPLIER * d * (d + 1) / eps**2))",
            "sample_size_multiplier": SAMPLE_SIZE_MULTIPLIER,
        }
        data_path = f"results_{{model}}_{values[0]}_{values[-1]}.npy"
        combined_path = f"dimension_sweep_{values[0]}_{values[-1]}_raw_error.pdf"
        separate_path = f"{{model}}_{values[0]}_{values[-1]}_raw_error.pdf"
        xlabel = "Dimension (d)"
    elif sweep == "epsilon":
        values = DEFAULT_EPS_LIST if values is None else values
        settings = [(FIXED_D, N, eps) for eps in values]
        metadata = {
            "fixed_d": FIXED_D, "eps_list": values, "n": N, "n_repeat": N_REPEAT,
            "sample_size_rule": "fixed N",
        }
        data_path = f"results_eps_sweep_{{model}}_d{FIXED_D}.npy"
        combined_path = f"epsilon_sweep_d{FIXED_D}_raw_error.pdf"
        separate_path = f"{{model}}_eps_d{FIXED_D}_raw_error.pdf"
        xlabel = r"Corruption Level ($\epsilon$)"
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    all_results = []
    for model_class, model_kwargs in CORRUPTION_MODELS:
        model = make_corruption_model(model_class, model_kwargs)
        model_name = model.__class__.__name__
        filename = data_path.format(model=model_name)
        if plot_only:
            all_results.append((model_name, np.load(filename, allow_pickle=True).item()))
            continue

        results = {
            "metadata": {
                **metadata,
                "model": model_name, "corruption_model_kwargs": model_kwargs,
                "algorithms": ALGO_LIST, "pairwise": not ZERO_MEAN, "zero_mean": ZERO_MEAN,
                "coeff_start": ADAPTIVE_COEFF_START, "coeff_min": ADAPTIVE_COEFF_MIN,
                "coeff_step": ADAPTIVE_COEFF_STEP, "seed": SEED,
            },
            "sampling_error": [], "naive_error": [],
            **{algo: {"raw": [], "excess": []} for algo in ALGO_LIST},
        }
        for d, n, eps in settings:
            print(f"\n{model_name}: d={d}, n={n}, eps={eps}, trials={N_REPEAT}")
            trials = run_trials(model, Params(d=d, n=n, eps=eps), N_REPEAT, ALGO_LIST, make_estimator)
            append_trials(results, trials, ALGO_LIST)
            np.save(filename, results)
        all_results.append((model_name, results))

    plot_sweep(all_results, values, xlabel, N_REPEAT, STYLES, MODEL_LABELS,
               combined_path, separate_path, combine=COMBINE_MODEL_PANELS)
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run and plot the synthetic covariance experiments.")
    sweeps = parser.add_subparsers(dest="sweep", required=True)
    dimension = sweeps.add_parser("dimension", help="Vary dimension at fixed corruption level.")
    dimension.add_argument("values", type=int, nargs="*", default=DEFAULT_DIMS, metavar="D")
    epsilon = sweeps.add_parser("epsilon", help="Vary corruption level at fixed dimension and sample size.")
    epsilon.add_argument("values", type=float, nargs="*", default=DEFAULT_EPS_LIST, metavar="EPS")
    for subparser in (dimension, epsilon):
        subparser.add_argument("--plot-only", action="store_true", help="Plot the saved results without fitting.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args.sweep, args.values, plot_only=args.plot_only)
