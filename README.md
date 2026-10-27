# Robust Gaussian Covariance Estimation via Nonconvex Optimization

Code for the synthetic Gaussian and semi-synthetic genomic experiments in the paper.

## Setup

The code has been checked with Python 3.14.6 and the versions in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the commands below from the repository root. A GPU is not required. The QUE subroutine uses CUDA when available and otherwise runs on the CPU.

## Synthetic Experiments

```bash
python synthetic_experiments.py dimension
python synthetic_experiments.py epsilon
```

- Dimension sweep: `d = 5, 10, 15, 20, 25, 30`, `eps = 0.10`, and `n = 2*d*(d+1)/eps**2`.
- Corruption sweep: `eps = 0.05, 0.10, 0.15, 0.20, 0.25, 0.30`, `d = 10`, and `n = 100000`.

Both use the cross-polytope, inverse-spectrum, and reversed-spectrum models, with 20 trials per configuration. Parameters are at the top of `synthetic_experiments.py`. The largest dimension uses 186,000 samples, so the full sweeps can take substantial time and memory.

Each command saves `results_*.npy` files and its corresponding PDF in the repository root. To redraw the figures from these saved results without rerunning the estimators:

```bash
python synthetic_experiments.py dimension --plot-only
python synthetic_experiments.py epsilon --plot-only
```

## Genomic Experiments

We use the public PCA files for the 1,387 POPRES samples from [Novembre et al. (2008)](https://github.com/NovembreLab/Novembre_etal_2008_misc), not the raw genotype data. Create a `genomicData/` directory in the repository root and place these four files directly inside it, keeping their names:

- [POPRES_08_24_01.EuroThinFinal.LD_0.8.exLD.out0-PCA.eigs](https://raw.githubusercontent.com/NovembreLab/Novembre_etal_2008_misc/master/files/POPRES_08_24_01.EuroThinFinal.LD_0.8.exLD.out0-PCA.eigs)
- [POPRES_08_24_01.EuroThinFinal.LD_0.8.exLD.out0-PCA.eval](https://raw.githubusercontent.com/NovembreLab/Novembre_etal_2008_misc/master/files/POPRES_08_24_01.EuroThinFinal.LD_0.8.exLD.out0-PCA.eval)
- [POPRESID_Color.txt](https://raw.githubusercontent.com/NovembreLab/Novembre_etal_2008_misc/master/files/POPRESID_Color.txt)
- [colors.txt](https://raw.githubusercontent.com/hoonose/robust-filter/master/genomicData/colors.txt)

The first three files come from the NovembreLab repository. The color map comes from the filtering repository below. The script constructs the 20-dimensional representation using the preprocessing convention of that implementation.

```bash
python genomic_experiments.py
```

This runs the discrete-block and inverse-spectrum corruptions at 5% and 10%, with 20 trials per configuration. The clean data stay fixed across trials. Outputs are saved in `figs - genomic/`:

- `DiscreteBlock_comparison.pdf` and `InverseSpectrum_comparison.pdf`: representative projections from the first trial at 5% corruption. Individual panels are also saved.
- `results_genomic_raw.json`: per-trial results.
- `results_genomic_summary.csv` and `genomic_results_table.tex`: means and standard errors.

Both experiment scripts initialize the random seed once per run to `1520761101`. Rerunning a command overwrites its existing outputs. Numerical results can differ slightly across library versions and CPU/GPU backends.

## Source Files

- `synthetic_experiments.py`, `genomic_experiments.py`: experiment and plotting scripts.
- `pgdCovAlgo.py`: PGD Covariance for the synthetic experiments.
- `pgdCovAlgo_topk.py`: the full top-three objective used in the genomic experiments.
- `filterCovAlgo.py`: the filtering baseline.
- `iterRefAlgo.py`: iterative refinement with PGD or QUE robust mean estimation.
- `corruption_models.py`: the three synthetic corruption models.

## References and Source Repositories

- [hoonose/robust-filter](https://github.com/hoonose/robust-filter), *Being Robust (in High Dimensions) Can Be Practical* (Diakonikolas et al., 2017): reference implementation for the filtering baseline and the genomic experiment setup.
- [twistedcubic/que-outlier-detection](https://github.com/twistedcubic/que-outlier-detection), *Quantum Entropy Scoring for Fast Robust Mean Estimation and Improved Outlier Detection* (Dong, Hopkins, and Li, 2019): reference for the practical score-and-remove QUE subroutine.
- [NovembreLab/Novembre_etal_2008_misc](https://github.com/NovembreLab/Novembre_etal_2008_misc) *Genes Mirror Geography in Europe* (Novembre et al., 2008): the preprocessed genomic data and sample color labels.

The iterative-refinement baselines are our implementations of the framework in [Cheng et al. (2019)](https://proceedings.mlr.press/v99/cheng19a.html), using either the PGD robust mean estimator of [Cheng et al. (2020)](https://proceedings.mlr.press/v119/cheng20a.html) or the QUE subroutine. Our modifications to the reference implementations and parameter choices are described in the paper's experimental appendix. 
