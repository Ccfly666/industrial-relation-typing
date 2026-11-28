# Industrial Relation Typing

Research implementation accompanying **Multi-source Concept Representation and Query-conditioned Relational Evidence for Fine-grained Semantic Relation Typing in Industrial Taxonomies** (manuscript; publication details pending).

Given a related, ordered concept pair `(source, target)`, predict its semantic scope relation. Disjoint pairs are outside this four-class task.

| Index | Label | Meaning in source-to-target direction |
| --- | --- | --- |
| 0 | `=` | Equal scopes |
| 1 | `>` | Source strictly contains target |
| 2 | `<` | Source is strictly contained in target |
| 3 | `~` | Nonempty intersection, with neither scope containing the other |

## Method and implementation

| Component | Coordinates | Information | Entry point |
| --- | ---: | --- | --- |
| A64 | 64 | Supervised low-rank text representation | `method.text_matrices`, `method.fit_a64`, `projection.solve` |
| B7 | 7 | Catalogue depths, depth differences and name similarities | `method.name_and_depth` |
| R1 | 12 | Admissible mapping degrees, availability and endpoint type proportions | `r1.relation_features` |
| Q63 | 63 | Query-conditioned type composition, mean similarity strength and supporting summaries | `q63_kernel.summarize`, `method.q63` |

AB has 71 active coordinates, ABR 83, and Full 146. The common MLP input always has 146 coordinates: removed channels are zeroed **after fit-only scaling**. R1 uses endpoint-local types (target types are inverted); Q63 uses the original source-to-target relation types on both sides. Feature construction uses the scoped mapping graph and excludes each training query's registered block.

| Task | Standards | Formal pairs |
| --- | --- | ---: |
| E | ETIM–ECLASS | 3,272 |
| C | ECLASS–GPC | 10,818 |
| U | ECLASS–UNSPSC | 100,320 |
| G | GPC–UNSPSC | 17,088 |

## Installation

```sh
git clone https://github.com/Ccfly666/industrial-relation-typing.git
cd industrial-relation-typing
git rev-parse HEAD
python -m venv .venv
```

Record the printed commit with your results. For the previously released research implementation, use `git checkout f61a24be2db923447017ccf705be8ebfae99acd7`; the current documentation and scope-suite wrapper are later usability additions. Activate with `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux/macOS. Use **Python 3.12** and choose one PyTorch build before installing the remaining dependencies:

```sh
# CPU (component checks and table generation)
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

```sh
# CUDA 11.8 reference build (Windows/Linux with a compatible NVIDIA driver)
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Check the installed build with `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`. The exact `torch==2.5.1` requirement accepts the selected `+cpu` or `+cu118` build. [Official version-specific instructions](https://pytorch.org/get-started/previous-versions/#v251) cover supported platforms. `requirements-lock.txt` records the historical CUDA environment; optional name encoding uses `requirements-features.txt`. Exact packages from the fresh CPU check are recorded in [requirements-cpu-tested.txt](requirements-cpu-tested.txt). See the [reproduction guide](docs/REPRODUCIBILITY.md) for tested environments and input schemas.

## Quick start: no benchmark download

```sh
python scripts/method.py smoke
python -m unittest discover -s tests -v
python scripts/generate_tables.py --input results/main_seed_summary.csv --out work/tables
python scripts/build_scope_suite.py --help
```

The smoke check uses synthetic inputs. Table generation reads the included aggregates and does not train a model.

## Data and model inputs

Benchmark RDF and encoder weights are obtained separately from their author sources. [DATA_ACCESS.md](DATA_ACCESS.md) lists public and retained inputs, source identities and access routes. The [reproduction guide](docs/REPRODUCIBILITY.md) separates rebuilding a benchmark run from verifying the reported retained predictions.

```sh
python scripts/fetch_sources.py --out work/benchmark.zip --sha256 04cc6dd2a2f8d173ddcc19822428fa9f505e3855840f06b19072306233d240eb --url https://zenodo.org/api/records/17091043/files/benchmark.zip/content --extract
python scripts/audit.py --input work/benchmark/etim-eclass --task E --out work/audit_E
```

## Reproducing experiments

After restoring the scoped labels and preparing endpoint name vectors, build five outer and fifteen inner feature files per task:

```sh
python scripts/build_scope_suite.py --task E --pool work/E/E__pool_public.npz --labels-dir work/E --name-vectors work/name_vectors.npz --out work/scopes --device cuda --dry-run
```

Inspect the commands, then remove `--dry-run` to execute. The wrapper calls `build_features.py` without changing feature calculations or data permissions. Train the desired family explicitly:

```sh
python scripts/train_classifier.py --suite work/scopes --task E --out work/E_budget80 --budget 80 --device cuda
python scripts/train_classifier.py --suite work/scopes --task E --out work/E_budget160 --budget 160 --device cuda
```

Repeat with C/U/G. Selection averages three inner raw Macro-F1 scores; final fitting uses three seeds. Outer-test labels enter only evaluation. Config JSON files describe experiment recipes; CLI arguments select executable settings.

## Results

Operational pooled OOF Macro-F1 (%), averaged over three seeds. Each epoch ceiling defines a separate experiment family.

| Task | Ceiling | AB | ABR | Full |
| --- | ---: | ---: | ---: | ---: |
| E | 80 | 78.03 | 89.70 | 91.69 |
| E | 160 | 83.95 | 91.50 | 92.37 |
| C | 80 | 57.78 | 74.03 | 75.67 |
| C | 160 | 67.29 | 74.47 | 75.94 |
| U | 80 | 75.39 | 84.44 | 86.10 |
| U | 160 | 75.73 | 84.45 | 86.10 |
| G | 80 | 62.19 | 73.99 | 76.68 |
| G | 160 | 68.94 | 74.26 | 76.62 |

[Results index](results/README.md) maps files, units, stable table keys and manuscript locations. It includes the seed summaries, all seven statistical families, mechanism comparisons, runtime and memory measurements.

## Repository structure

| Directory | Contents |
| --- | --- |
| `scripts/` | Data preparation, features, training, evaluation, statistics and plotting |
| `configs/` | Method and 80/160-epoch recipe descriptions |
| `manifests/` | Source hashes, retained splits and replay specifications |
| `results/` | Aggregate scores, ablations, intervals and costs |
| `tests/` | Synthetic component and orchestration tests |
| `docs/REPRODUCIBILITY.md` | Commands, schemas and tested reproduction scope |

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the software and record the immutable commit used. Manuscript bibliographic details will be added when available.

Project-owned code is released under the [MIT License](LICENSE). Benchmark and model terms are described in [DATA_ACCESS.md](DATA_ACCESS.md) and [third-party notices](docs/THIRD_PARTY_LICENSES.md).
