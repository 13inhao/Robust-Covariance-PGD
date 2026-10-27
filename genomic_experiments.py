"""Run the POPRES experiments and save projections, raw results, and tables."""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ortho_group

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgdCovAlgo_topk import (
    RobustCovariancePGD_TopK as PGD_TopK,
    error_Mahalanobis,
    error_Mahalanobis_spectral,
    top2_subspace_error,
)
from filterCovAlgo import FilterCovariance as Filtering
from iterRefAlgo import IterativeRefinement as IterRef

RANDOM_SEED = 1520761101
EPS_VALUES = (0.05, 0.10)
N_REPEAT = 20
DIMENSION = 20
CORRUPTIONS = ("discrete_block", "inverse_spectrum")
OUTPUT_DIR = Path("figs - genomic")
ALGORITHM_ORDER = ("Corrupted PCA", "PGD Covariance", "Filtering", "IterRef PGD", "IterRef QUE")


class GenomicDiscreteBlockModel:
    def __init__(self, scale=1.0 / 24.0):
        self.scale = scale

    def generate(self, X_clean, eps):
        n_clean, d = X_clean.shape
        n_noise = int(round(eps * n_clean / (1.0 - eps)))
        noise_part1 = np.random.randint(0, 3, size=(n_noise, d // 2))
        noise_part2 = np.random.randint(2, 4, size=(n_noise, d // 2))
        X_noise = np.hstack([noise_part1, noise_part2]) * self.scale

        Q = ortho_group.rvs(d)
        X_clean_rot, X_noise_rot = X_clean @ Q, X_noise @ Q
        return np.vstack([X_clean_rot, X_noise_rot]), X_clean_rot, X_noise_rot, np.zeros((n_noise, 3))


class GenomicInverseSpectrumModel:
    def __init__(self, variance_multiplier=100.0):
        self.variance_multiplier = variance_multiplier

    def generate(self, X_clean, eps):
        n_clean, d = X_clean.shape
        n_noise = int(round(eps * n_clean / (1.0 - eps)))
        evals, evecs = np.linalg.eigh(np.cov(X_clean, rowvar=False))
        v_min, lambda_max = evecs[:, 0], evals[-1]
        mu = np.mean(X_clean, axis=0)
        outlier_std = np.sqrt(lambda_max * self.variance_multiplier)
        X_noise = np.outer(np.random.normal(0, outlier_std, n_noise), v_min)
        X_noise += mu

        Q = ortho_group.rvs(d)
        X_clean_rot, X_noise_rot = X_clean @ Q, X_noise @ Q
        return np.vstack([X_clean_rot, X_noise_rot]), X_clean_rot, X_noise_rot, np.zeros((n_noise, 3))


def load_clean_popres(d=20, data_dir="genomicData"):
    base = Path(data_dir)
    eigs_file = base / "POPRES_08_24_01.EuroThinFinal.LD_0.8.exLD.out0-PCA.eigs"
    eval_file = base / "POPRES_08_24_01.EuroThinFinal.LD_0.8.exLD.out0-PCA.eval"
    df_eigs = pd.read_csv(eigs_file, sep=r"\s+", engine="python", skiprows=1, header=None)
    V = df_eigs.iloc[:, 2:2 + d].values
    evals = np.loadtxt(eval_file)[:d]
    # Keep the scaling used by the original MATLAB pipeline.
    X_clean = V * evals

    color_defs = pd.read_csv(base / "colors.txt", sep="\t", index_col=0, header=None)
    color_dict = {name: vals.values / 255.0 for name, vals in color_defs.iterrows()}
    labels = pd.read_csv(base / "POPRESID_Color.txt", sep=r"\s+", engine="python", header=None)
    C_clean = np.array([color_dict[name] for name in labels.iloc[:, 1]])
    return X_clean, C_clean


def project_top2(covariance, X_clean, X_noise):
    _, _, Vt = np.linalg.svd(covariance)
    clean, noise = X_clean @ Vt[:2].T, X_noise @ Vt[:2].T
    signs = np.sign(clean[np.argmax(np.abs(clean), axis=0), range(2)])
    clean *= signs
    noise *= signs
    return np.column_stack((clean[:, 1], -clean[:, 0])), np.column_stack((noise[:, 1], -noise[:, 0]))


def visualize_projection(covariance, X_clean, X_noise, colors, title, path, show_noise=True):
    clean, noise = project_top2(covariance, X_clean, X_noise)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(clean[:, 0], clean[:, 1], c=colors, s=10, alpha=0.8)
    if show_noise:
        ax.scatter(noise[:, 0], noise[:, 1], c="black", s=10, label="Noise")
    ax.set_title(title)
    if show_noise:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def visualize_comparison(D, X_clean_rot, X_noise_rot, C_clean, Sigma_clean,
                         estimates, model_name, output_dir):
    panels = [("Clean", Sigma_clean, False), ("Corrupted PCA", np.cov(D, rowvar=False), True)]
    panels.extend((name, Sigma_hat, True) for name, Sigma_hat in estimates.items())
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.3))
    for ax, (name, covariance, show_noise) in zip(axes.flat, panels):
        clean, noise = project_top2(covariance, X_clean_rot, X_noise_rot)
        ax.scatter(clean[:, 0], clean[:, 1], c=C_clean, s=5.5, alpha=0.8)
        if show_noise:
            ax.scatter(noise[:, 0], noise[:, 1], c="black", s=6)
        error = top2_subspace_error(covariance, Sigma_clean)
        ax.set_title(f"{name} ({error:.3f})", fontsize=11, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.025, top=0.985, wspace=0.08, hspace=0.16)
    fig.savefig(Path(output_dir) / f"{model_name}_comparison.pdf",
                format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def covariance_errors(estimate, reference, runtime=0.0):
    return {
        "fro": float(error_Mahalanobis(estimate, reference)),
        "spectral": float(error_Mahalanobis_spectral(estimate, reference)),
        "top2": float(top2_subspace_error(estimate, reference)),
        "runtime": runtime,
    }


def run_genomic_experiment(seed=RANDOM_SEED, corruption="inverse_spectrum", eps_val=EPS_VALUES[0],
                           d_val=DIMENSION, output_dir=OUTPUT_DIR, make_figures=True, verbose=True):
    if seed is not None:
        np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_clean, C_clean = load_clean_popres(d=d_val)
    if corruption == "discrete_block":
        noise_model = GenomicDiscreteBlockModel(scale=1.0 / 24.0)
        model_name = "DiscreteBlock"
    elif corruption == "inverse_spectrum":
        noise_model = GenomicInverseSpectrumModel(variance_multiplier=100.0)
        model_name = "InverseSpectrum"
    else:
        raise ValueError(f"unknown corruption model: {corruption!r}")

    D, X_clean_rot, X_noise_rot, _ = noise_model.generate(X_clean, eps_val)
    Sigma_clean_emp = np.cov(X_clean_rot, rowvar=False)
    Sigma_corrupted = np.cov(D, rowvar=False)
    exp_prefix = str(output_dir / model_name)
    if make_figures:
        visualize_projection(Sigma_clean_emp, X_clean_rot, X_noise_rot, C_clean,
                             "Original Clean Data Projection", f"{exp_prefix}_clean_plot.pdf", False)
        visualize_projection(Sigma_corrupted, X_clean_rot, X_noise_rot, C_clean,
                             "Standard PCA Projection", f"{exp_prefix}_standard_pca_plot.pdf")

    n_samples, d_dim = D.shape
    estimators = [
        ("PGD Covariance", PGD_TopK(d_dim, eps_val, pairwise=False)),
        ("Filtering", Filtering(d_dim, eps_val, pairwise=False)),
        ("IterRef PGD", IterRef(d_dim, eps_val, rme_subroutine="PGD", pairwise=False)),
        ("IterRef QUE", IterRef(d_dim, eps_val, rme_subroutine="QUE", pairwise=False)),
    ]
    results = {"Corrupted PCA": covariance_errors(Sigma_corrupted, Sigma_clean_emp)}
    covariance_estimates = {}
    for algo_name, estimator in estimators:
        start = time.time()
        try:
            Sigma_hat = estimator.robustCovEstimator(D)
            runtime = time.time() - start
            results[algo_name] = covariance_errors(Sigma_hat, Sigma_clean_emp, runtime)
            covariance_estimates[algo_name] = Sigma_hat
            if verbose:
                print(f"{algo_name}: top2={results[algo_name]['top2']:.6f}, time={runtime:.2f}s")
            if make_figures:
                filename = f"{exp_prefix}_{algo_name.replace(' ', '_')}_plot.pdf"
                visualize_projection(Sigma_hat, X_clean_rot, X_noise_rot, C_clean,
                                     f"{algo_name} Projection", filename)
        except Exception as e:
            results[algo_name] = {"runtime": time.time() - start, "error": str(e)}
            if verbose:
                print(f"{algo_name} failed: {e}")

    if make_figures:
        visualize_comparison(D, X_clean_rot, X_noise_rot, C_clean, Sigma_clean_emp,
                             covariance_estimates, model_name, output_dir)
    run_info = {
        "seed": int(seed) if seed is not None else None,
        "epsilon": float(eps_val),
        "dimension": int(d_dim),
        "n_clean": int(X_clean.shape[0]),
        "n_outliers": int(X_noise_rot.shape[0]),
        "n_total": int(n_samples),
        "pairwise": False,
        "inverse_variance_multiplier": 100.0 if corruption == "inverse_spectrum" else None,
    }
    return model_name, results, run_info


def summarize_runs(runs, eps_values=EPS_VALUES):
    rows = []
    for model_name in ("DiscreteBlock", "InverseSpectrum"):
        for eps_val in eps_values:
            setting_runs = [run for run in runs
                            if run["model"] == model_name and np.isclose(run["epsilon"], eps_val)]
            if not setting_runs:
                continue
            for algorithm in ALGORITHM_ORDER:
                entries = [run["results"].get(algorithm, {}) for run in setting_runs]
                values = np.asarray([entry["top2"] for entry in entries if entry.get("top2") is not None], dtype=float)
                runtimes = np.asarray([entry["runtime"] for entry in entries if entry.get("runtime") is not None], dtype=float)
                mean = float(values.mean()) if values.size else np.nan
                standard_error = np.nan
                if values.size:
                    standard_error = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
                rows.append({
                    "model": model_name, "epsilon": float(eps_val),
                    "n_clean": setting_runs[0]["n_clean"],
                    "n_outliers": setting_runs[0]["n_outliers"],
                    "n_total": setting_runs[0]["n_total"],
                    "algorithm": algorithm, "mean_top2_error": mean,
                    "standard_error": standard_error,
                    "mean_runtime_seconds": float(runtimes.mean()) if runtimes.size else np.nan,
                    "n_success": int(values.size), "n_trials": len(setting_runs),
                })
    return pd.DataFrame(rows)


def write_latex_table(summary, output_path):
    n_trials = int(summary["n_trials"].max())
    n_clean = int(summary["n_clean"].iloc[0])
    eps_values = sorted(summary["epsilon"].unique())
    sample_sizes = {float(row["epsilon"]): int(row["n_total"])
                    for _, row in summary.drop_duplicates("epsilon").iterrows()}
    model_labels = {"DiscreteBlock": "Discrete block", "InverseSpectrum": "Inverse spectrum"}
    columns = [(model, eps) for model in model_labels for eps in eps_values]
    sample_size_text = " and ".join(
        f"${sample_sizes[eps]}$ at ${100 * eps:g}\\%$ corruption" for eps in eps_values)
    n_eps = len(eps_values)
    header = " & ".join(f"\\multicolumn{{{n_eps}}}{{c}}{{{label}}}" for label in model_labels.values())
    levels = " & ".join(f"${100 * eps:g}\\%$" for _, eps in columns)
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{@{}l" + "c" * len(columns) + "@{}}",
        "\\toprule", " & " + header + " \\\\",
        f"\\cmidrule(lr){{2-{1 + n_eps}}} \\cmidrule(lr){{{2 + n_eps}-{1 + 2 * n_eps}}}",
        "Estimator & " + levels + " \\\\", "\\midrule",
    ]
    for algorithm in ALGORITHM_ORDER:
        cells = []
        for model_name, eps_val in columns:
            match = summary[(summary["model"] == model_name)
                            & np.isclose(summary["epsilon"], eps_val)
                            & (summary["algorithm"] == algorithm)]
            if match.empty or not np.isfinite(match.iloc[0]["mean_top2_error"]):
                cells.append("--")
            else:
                row = match.iloc[0]
                cells.append(f"${row['mean_top2_error']:.4f} \\pm {row['standard_error']:.4f}$")
        name = "Standard PCA" if algorithm == "Corrupted PCA" else f"\\texttt{{{algorithm}}}"
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule", "\\end{tabular}",
        f"\\caption{{Top-two subspace error on the POPRES experiments. The clean sample size is ${n_clean}$, and the total sample sizes are {sample_size_text}. Entries report the mean and one standard error over ${n_trials}$ trials.}}",
        "\\label{tab:genomic_results}", "\\end{table}",
    ])
    Path(output_path).write_text("\n".join(lines) + "\n")


def run_genomic_sweep(n_repeat=N_REPEAT, eps_values=EPS_VALUES, output_dir=OUTPUT_DIR,
                      save_representative_figures=True, verbose=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    runs = []
    for corruption_name in CORRUPTIONS:
        for eps_val in eps_values:
            for trial in range(n_repeat):
                print(f"[{corruption_name} | eps={eps_val:.2f} | trial={trial + 1}/{n_repeat}]")
                make_figures = save_representative_figures and trial == 0 and np.isclose(eps_val, eps_values[0])
                model_name, results, run_info = run_genomic_experiment(
                    seed=None, corruption=corruption_name, eps_val=eps_val, d_val=DIMENSION,
                    output_dir=output_dir, make_figures=make_figures, verbose=verbose)
                runs.append({"model": model_name, "global_seed": RANDOM_SEED, "trial": trial + 1,
                             **run_info, "results": results})

    summary = summarize_runs(runs, eps_values=eps_values)
    with (output_dir / "results_genomic_raw.json").open("w") as f:
        json.dump(runs, f, indent=2)
    summary.to_csv(output_dir / "results_genomic_summary.csv", index=False)
    write_latex_table(summary, output_dir / "genomic_results_table.tex")
    return runs, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=N_REPEAT)
    parser.add_argument("--eps", type=float, nargs="+", default=list(EPS_VALUES))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _, summary_df = run_genomic_sweep(n_repeat=args.trials, eps_values=tuple(args.eps),
                                     output_dir=args.output_dir,
                                     save_representative_figures=not args.no_figures, verbose=args.verbose)
    print("\nTop-two subspace error (mean +/- standard error):")
    for _, row in summary_df.iterrows():
        print(f"{row['model']:>15} | eps={row['epsilon']:.2f} | {row['algorithm']:<16} | "
              f"{row['mean_top2_error']:.4f} +/- {row['standard_error']:.4f} | "
              f"runtime={row['mean_runtime_seconds']:.2f}s")
    print(f"\nSaved figures and metrics to: {args.output_dir.resolve()}")
