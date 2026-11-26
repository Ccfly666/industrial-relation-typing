# Industrial Relation Typing

Official implementation of multi-source concept representation and query-conditioned relational evidence for fine-grained semantic relation typing in industrial taxonomies.

The task assigns one of four relations to an **ordered pair of already related concepts**: equivalence (`=`), strict source containment (`>`), strict target containment (`<`), or strict overlap (`~`). It does not perform unrestricted candidate discovery or add an unrelated class.

| Task | Standards | Formal pairs |
| --- | --- | ---: |
| E | ETIM–ECLASS | 3,272 |
| C | ECLASS–GPC | 10,818 |
| U | ECLASS–UNSPSC | 100,320 |
| G | GPC–UNSPSC | 17,088 |

E uses the corrected proof-group split. The retained metadata defines five outer folds and three inner folds. Historical development exposure and retained preprocessing decisions are documented in [the limitations](docs/LIMITATIONS.md).

## Method

- **A64:** four ordered name/context fields represented by a supervised 64-dimensional low-rank text projection.
- **B7:** seven public catalogue-position and name-similarity features.
- **R1:** twelve permitted mapping-neighbourhood features, built from the legal fit graph with query dependency-group exclusion.
- **Q63:** sixty-three query-conditioned summaries of permitted relation events; these are derived coordinates, not sixty-three independent information sources.

The common MLP layout has 146 coordinates and a `146 → 128 → 64 → 4` ReLU head with dropout 0.1. AB uses the first 71 coordinates, ABR the first 83, and Full all 146. Disabled coordinates are zeroed **after** scaling. Operational predictions copy the same fit-specific fixed-LR AB probabilities when neither endpoint has relation evidence. Raw and operational probabilities remain distinct.

## Installation

Use Python 3.12. The numerical reference uses PyTorch 2.5.1 with CUDA 11.8 and NumPy 2.1.3. A CPU installation can run component checks, but is not a claim of numerically identical CUDA training.

```sh
python -m venv .venv
```

Activate `.venv` with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS, then install:

```sh
python -m pip install -r requirements.txt
python scripts/method.py smoke
python -m unittest discover -s tests -v
```

For the recorded reference package versions, use `requirements-lock.txt` instead of `requirements.txt`. `requirements-features.txt` lists the optional text-encoding dependencies. The original independent acceptance environment used physically copied and checksum-verified PyTorch/NumPy distributions; a fresh installation from the lock file has not been separately certified on every platform.

## Reproduction workflow

### 1. Inspect aggregate results without restricted assets

```sh
python scripts/generate_tables.py --input results/main_seed_summary.csv --out work/tables
```

The CSV files under [`results/`](results/) contain reported aggregate outcomes, uncertainty summaries, budget diagnostics, and provenance registries. `main_seed_summary.csv` describes its frozen 80-budget family; the `budget_*` files describe the separate 160-budget analysis. Do not combine different families or substitute a selected seed for a three-seed mean.

### 2. Obtain the source benchmark legally

Read [DATA_ACCESS.md](DATA_ACCESS.md) before downloading. The repository does not redistribute the industrial taxonomies, reference relations, pretrained weights, embeddings, or individual predictions.

```sh
python scripts/fetch_sources.py --out work/benchmark.zip --sha256 04cc6dd2a2f8d173ddcc19822428fa9f505e3855840f06b19072306233d240eb --url https://zenodo.org/api/records/17091043/files/benchmark.zip/content --extract
python scripts/audit.py --input work/benchmark/etim-eclass --task E --out work/audit_E
```

The downloader also accepts `--local PATH` instead of `--url URL` for an authorized snapshot. Checksums identify the declared release; access does not grant redistribution rights. Inspect the downloaded archive layout and adjust the dataset directory if it has an additional enclosing directory.

### 3. Restore the declared formal identities and features

The following commands illustrate outer fold 0 of E. `NAME_VECTORS.npz` must be an authorized cache with embeddings in the exact sorted snapshot/URI order; `scripts/encode_names.py --help` describes the frozen ONNX encoder entry. The public split metadata contains reference-cell indices and group hashes, not taxonomy text or relation labels.

```sh
python scripts/verify_dataset.py --input work/benchmark/etim-eclass --manifest manifests/E_retained_roles_splits.csv --task E --name-vectors NAME_VECTORS.npz --out work/E
python scripts/build_features.py --pool work/E/E__pool_public.npz --fit-labels work/E/E__o0__fit_labels.npz --name-vectors NAME_VECTORS.npz --outer 0 --out work/scopes/E_o0.npz --device cuda
```

An explicit `--projector PROJECTOR.npz` reuses a retained **fit-specific** A64 projection. Without it, the tool fits a new supervised projection on fit labels only. Strict historical numerical comparison depends on the declared projector, vectors, batch ordering, and numerical backend. This is different from claiming that all upstream assets were independently regenerated.

For an inner scope, also pass `--inner 0` and the matching `--validation-labels work/E/E__o0i0__validation_labels.npz`, with its matching inner fit-label file. Only inner validation files contain `y_eval` for model selection; sealed outer labels are used only in subsequent evaluation. Never supply outer-test labels to feature fitting or selection.

### 4. Run a selected fold or the complete nested head suite

The accepted E fold-0 Full reference used seed 2026111601 and the 160-budget selected recipe `lr=0.001, epochs=140`:

```sh
python scripts/train_classifier.py --input work/scopes/E_o0.npz --out work/E_o0_Full --arm Full --lr 0.001 --epochs 140 --seed 2026111601 --device cuda
```

Prepare all twenty scoped inputs for each task: five outer scopes (`E_o0.npz` through `E_o4.npz`) and fifteen inner scopes (`E_o0i0.npz`, etc.). Each scope must bind its own legal common fallback. Then run the desired separate budget family:

```sh
python scripts/train_classifier.py --suite work/scopes --task E --out work/E_budget80 --budget 80 --device cuda
python scripts/train_classifier.py --suite work/scopes --task E --out work/E_budget160 --budget 160 --device cuda
```

Repeat for C/U/G with the corresponding scoped files. The suite performs complete inner candidate trajectories without patience truncation, selects by the mean of three inner raw Macro-F1 values, and fits three final seeds for each of AB/ABR/Full. These commands perform substantial training; they are not part of the smoke test. The JSON files under [`configs/`](configs/) document the recipes; the executable selects 80 or 160 through the explicit `--budget` argument, rather than consuming an arbitrary configuration file.

### 5. Evaluate and replay reported statistics

```sh
python scripts/evaluate_oof.py --help
python scripts/statistical_analysis.py --help
python scripts/verify_reported_results.py --help
```

Use the declared gold row identities and aggregate all five outer prediction files before computing the four-class pooled OOF Macro-F1. Summarize the three per-seed pooled scores; a fold-score average or probability ensemble is a different quantity. Report raw, operational, nonempty-evidence, and zero-evidence results separately. Seed standard deviation is not a confidence interval.

Authorized holders of the retained prediction files can run:

```sh
python scripts/method.py check-scores AUTHORIZED_PREDICTION_FOLDER
python scripts/verify_reported_results.py --stat-assets AUTHORIZED_STATISTICAL_BLOBS --out work/original54_check.csv
```

The required private assets are identified by hashes in the public manifests and are not supplied by this repository. The original 54 intervals retain their original comparison families; later budget comparisons have separate families. See [the actual acceptance scope](docs/REPRODUCTION.md).

## What has been validated

| Level | Evidence and boundary |
| --- | --- |
| Saved results | Retained main probabilities/scores and 54 original interval calculations were checked with explicitly supplied authorized assets. |
| Runnable components | Acquisition/parse, identity restoration, core features, classifier and statistics entries, tables, and component tests were exercised. |
| Specified E scope | E outer fold 0, seed 2026111601, Full, selected 160-budget recipe was rebuilt and evaluated in a separate environment with declared retained inputs. Prediction labels agreed; small probability/backend differences are retained in the acceptance records. |
| All four tasks from scratch | **Not independently completed.** Providing a suite entry does not certify a full independent rerun. |
| All historical comparison families | **Not independently retrained.** Historical role decisions and some upstream assets remain explicit retained dependencies. |

Detailed numerical differences and dependency limits are in [REPRODUCTION.md](docs/REPRODUCTION.md) and [LIMITATIONS.md](docs/LIMITATIONS.md). Stored acceptance JSON files record the original local tests, not a guarantee for every new machine.

## Repository guide

| Location | Contents |
| --- | --- |
| `scripts/` | Core feature kernels, acquisition/parse, model training, evaluation, statistical and plotting tools |
| `configs/` | Numerical and model recipes, including separate 80/160 budgets |
| `manifests/` | Source hashes, retained split/dependency metadata, code origins and release-file checksums |
| `results/` | Aggregate experiment results and evidence registries |
| `tests/` | Functional component tests |
| `docs/` | Reproduction limits and third-party notices |
| `verification/` | Sanitized historical acceptance receipts and small reference table/figure artifacts |

See [PROVENANCE.md](PROVENANCE.md), [CHANGELOG.md](CHANGELOG.md), and [CITATION.cff](CITATION.cff). Research software is released under the [MIT License](LICENSE). This license grants no rights to third-party taxonomies, model parameters, or separately obtained datasets; consult [third-party notices](docs/THIRD_PARTY_LICENSES.md).
