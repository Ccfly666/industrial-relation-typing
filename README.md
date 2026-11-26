# Industrial Relation Typing

## Overview

Code for fine-grained semantic relation typing between related concepts in industrial taxonomies. Given an ordered concept pair, the model predicts equivalence (`=`), strict source containment (`>`), strict target containment (`<`), or strict overlap (`~`).

The method combines **A64** supervised low-rank text features, **B7** catalogue-position and name-similarity features, **R1** mapping-neighbourhood profiles, and **Q63** query-conditioned relation summaries. AB, ABR and Full use a common 146-coordinate MLP layout with unused coordinates zeroed after scaling.

| Task | Standards | Pairs |
| --- | --- | ---: |
| E | ETIM–ECLASS | 3,272 |
| C | ECLASS–GPC | 10,818 |
| U | ECLASS–UNSPSC | 100,320 |
| G | GPC–UNSPSC | 17,088 |

## Repository structure

| Directory | Contents |
| --- | --- |
| `scripts/` | Data preparation, feature construction, training, evaluation, statistics and plotting |
| `configs/` | Method settings and 80/160-epoch experiment settings |
| `manifests/` | Source checksums, retained row/group splits and replay specifications |
| `results/` | Per-seed summaries, ablations, intervals and computation costs |
| `tests/` | Component tests |
| `docs/REPRODUCIBILITY.md` | Detailed commands, input schemas and tested workflows |

## Installation

Use Python 3.12. The reference environment uses PyTorch 2.5.1/CUDA 11.8 and NumPy 2.1.3.

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux/macOS, then:

```sh
python -m pip install -r requirements.txt
```

`requirements-lock.txt` records the reference package versions. Name encoding additionally uses `requirements-features.txt`.

## Data preparation

Download the benchmark from [Zenodo](https://doi.org/10.5281/zenodo.17091043), subject to its data terms:

```sh
python scripts/fetch_sources.py --out work/benchmark.zip --sha256 04cc6dd2a2f8d173ddcc19822428fa9f505e3855840f06b19072306233d240eb --url https://zenodo.org/api/records/17091043/files/benchmark.zip/content --extract
python scripts/audit.py --input work/benchmark/etim-eclass --task E --out work/audit_E
```

[DATA_ACCESS.md](DATA_ACCESS.md) describes the source snapshots, encoders and retained inputs. The [reproduction guide](docs/REPRODUCIBILITY.md) gives the commands for restoring row/group identities and building features.

## Quick start

These examples use synthetic inputs or the included aggregate results:

```sh
python scripts/method.py smoke
python -m unittest discover -s tests -v
python scripts/generate_tables.py --input results/main_seed_summary.csv --out work/tables
```

## Reproducing experiments

Prepare five outer and fifteen inner feature files for each task, with their scoped labels and common LR fallback. Run the selected budget:

```sh
python scripts/train_classifier.py --suite work/scopes --task E --out work/E_budget80 --budget 80 --device cuda
python scripts/train_classifier.py --suite work/scopes --task E --out work/E_budget160 --budget 160 --device cuda
```

Repeat with C/U/G. Both settings use three inner folds for selection and three final seeds. Outer-test labels are supplied only to evaluation. File schemas, one-fold execution, statistics and tested reproduction scope are described in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Results

Main 80-epoch results (three-seed pooled OOF Macro-F1, %):

| Task | AB | ABR | Full |
| --- | ---: | ---: | ---: |
| E | 78.03 | 89.70 | 91.69 |
| C | 57.78 | 74.03 | 75.67 |
| U | 75.39 | 84.44 | 86.10 |
| G | 62.19 | 73.99 | 76.68 |

| Results | File |
| --- | --- |
| Main 80-epoch family, three-seed summaries | [main_seed_summary.csv](results/main_seed_summary.csv) |
| 80/160 budget comparison | [budget_main_results.csv](results/budget_main_results.csv) |
| Main and mechanism intervals | [main_intervals.csv](results/main_intervals.csv), [mechanism_intervals.csv](results/mechanism_intervals.csv) |
| Budget intervals | [budget_statistical_intervals.csv](results/budget_statistical_intervals.csv) |
| All reported table cells | [paper_tables.csv](results/paper_tables.csv) |
| Timing and memory | [inference_timing_summary.csv](results/inference_timing_summary.csv), [memory_summary.csv](results/memory_summary.csv) |

Scores are four-class pooled OOF Macro-F1, summarized across three seeds. The 80- and 160-epoch settings are separate experiment families; their per-family statistical definitions are listed in [experiment_family_registry.csv](results/experiment_family_registry.csv).

## Citation

Use [CITATION.cff](CITATION.cff) to cite the software, and record the immutable commit used in an experiment. Manuscript bibliographic details will be added when available.

## License

Project code is released under the [MIT License](LICENSE). Benchmark and model access terms are listed in [DATA_ACCESS.md](DATA_ACCESS.md) and [third-party notices](docs/THIRD_PARTY_LICENSES.md).
