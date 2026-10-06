# kac-geometry — Large-deviation geometry of Kac-type kinetic models

Code and data accompanying

> T. A. Tazhimov, A. R. Sharden, V. S. Zheltukhin,
> *Large-deviation geometry of Kac-type kinetic models: cosh gradient structures, spectra, and a universal nonlinear correction to entropy dissipation*, submitted to Journal of Statistical Physics (2026).

Python (NumPy, SciPy, SymPy, Matplotlib). Units: collision rate λ = 1. Run every script from inside `scripts/`; outputs go to `../data/` and `../figures/`.

## Scripts and what they produce

| Script | Content | Paper |
|---|---|---|
| `kac_geom.py` | library + checks for the Kac model: grid, collision maps, Q, geometric-mean weight W, cosh potential Ψ*, Onsager operator K, Hermite spectrum, dissipations along the Krook–Wu solution, κ_eff, trajectory length | Secs. 3–4, 10, Fig. 1 |
| `kac_curv.py` | induced metric on the (a₄,a₆) moment submanifold, Gaussian curvature (Brioschi), geodesics by shooting vs linearized flow | Sec. 11, Fig. 3, Table 1 |
| `figs.py` | figures for the Kac model | Figs. 1–3 |
| `models.py` | inelastic Kac model (cooling law, symmetry defect, cosh-structure defect) and Carleman closed forms | Secs. 8, 13, Fig. 2 (right), Fig. 4 |
| `three_body.py`, `tb_traj.py` | three-body Kac model: exact SO(3) quadrature, spectrum, metric/curvature on (a₄,a₆), κ_eff along the exact moment trajectory | Secs. 4, 6, 11 |
| `c_coeff.py` | exact first-order coefficient c = ½⟨h³⟩ for m = 2, 3 (quadrature) and m = 4 (Monte Carlo over SO(4)) | Theorem 8.1 |
| `closeA.py`, `closeA2.py` | divergence of κ_eff at the positivity boundary (adaptive quadrature), self-similar series Ψ_m, global-contraction ratios, thermostat spectrum, grazing limit | Secs. 7, 10, 11 |
| `cbeta.py` | exact first-order coefficient for velocity-dependent kernels \|v−v*\|^β | Sec. 11, Fig. 5 |
| `figA5.py` | figure for Secs. 7, 10, 11 | Fig. 5 |

`data/` contains the JSON/NPY outputs of the original runs so that every table in the paper can be checked without recomputation.

## Reproduce
```bash
pip install -r requirements.txt
cd scripts
python kac_geom.py && python kac_curv.py && python figs.py      # Kac model (≈10 min)
python models.py                                                # inelastic + Carleman (≈5 min)
python three_body.py && python tb_traj.py                       # three-body (≈15 min)
python c_coeff.py && python cbeta.py                            # universal coefficient (≈3 min)
python closeA.py && python closeA2.py && python figA5.py        # boundary divergence, self-similar series (≈10 min)
```

## License
MIT. Please cite the paper and the Zenodo DOI of this repository.
