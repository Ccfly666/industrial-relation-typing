# Reproducing the experiments

## Environment and inputs

The tested environment uses Python 3.12.14, PyTorch 2.5.1+cu118, NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1 and Matplotlib 3.10.0. The reference classifier runs in CUDA float32 with TF32 and AMP disabled. Package versions are recorded in `requirements-lock.txt`. The tested separate environment used checksum-verified local PyTorch/NumPy distributions and separately installed numerical dependencies.

Obtain the RDF snapshots and encoder files as described in [DATA_ACCESS.md](../DATA_ACCESS.md). The workflow below uses task E; replace the task code, source directory and split CSV for C/U/G.

## Restore rows and splits

First build the pool and legal fit-label files:

```sh
python scripts/verify_dataset.py --input work/benchmark/etim-eclass --manifest manifests/E_retained_roles_splits.csv --task E --out work/E
```

The pool stores row/group identities, outer/inner assignments, endpoint keys and text fields. The endpoint keys are sorted snapshot-SHA-plus-URI identities. Prepare names in that exact order, encode each side and assemble the vector cache:

```sh
python scripts/prepare_name_vectors.py prepare --pool work/E/E__pool_public.npz --source-rdf work/benchmark/etim-eclass/source.rdf --target-rdf work/benchmark/etim-eclass/target.rdf --out work/names
python scripts/encode_names.py --names work/names/source_names.json --model MODEL.onnx --tokenizer TOKENIZER.json --out work/source_vectors.npz
python scripts/encode_names.py --names work/names/target_names.json --model MODEL.onnx --tokenizer TOKENIZER.json --out work/target_vectors.npz
python scripts/prepare_name_vectors.py pack --pool work/E/E__pool_public.npz --names-dir work/names --source-vectors work/source_vectors.npz --target-vectors work/target_vectors.npz --out work/name_vectors.npz
```

The preparation step checks the RDF snapshot identities; assembly checks encoded-name order and vector shapes. `work/name_vectors.npz` contains the two 384-dimensional endpoint arrays and their keys. Exact retained-vector replay also uses the original quantized batch grouping. In the following commands, `NAME_VECTORS.npz` denotes this assembled file or a matching retained cache.

Re-run pool construction with the vector cache to include B7's name cosine:

```sh
python scripts/verify_dataset.py --input work/benchmark/etim-eclass --manifest manifests/E_retained_roles_splits.csv --task E --name-vectors NAME_VECTORS.npz --out work/E
```

`E__outer_truth.npz` is the sealed evaluation file. Per-scope fit labels and inner validation labels are emitted separately. Retained split/role choices are inputs to this reconstruction, including the corrected E proof grouping and historical data-development exposure.

## Build A64, B7, R1 and Q63

```sh
python scripts/build_features.py --pool work/E/E__pool_public.npz --fit-labels work/E/E__o0__fit_labels.npz --name-vectors NAME_VECTORS.npz --outer 0 --out work/scopes/E_o0.npz --device cuda
python scripts/build_features.py --pool work/E/E__pool_public.npz --fit-labels work/E/E__o0i0__fit_labels.npz --validation-labels work/E/E__o0i0__validation_labels.npz --name-vectors NAME_VECTORS.npz --outer 0 --inner 0 --out work/scopes/E_o0i0.npz --device cuda
```

Repeat for all five outer and fifteen inner scopes. An explicit `--projector PROJECTOR.npz` reuses a retained fit-specific A64 projection; otherwise a new projection is fitted using only that scope's fit labels. Projectors contain `fit_rows`, `L` and `mean`. Vocabulary/text transforms, scalers and the common LR fallback are fitted on the matching fit range. Relation features use its legal graph and exclude the query dependency block. The fixed hash prefix in the method settings is retained to reproduce the original block ordering.

Each prepared scope contains `x_fit`/`x_eval` with 146 standardized coordinates, `y_fit`, `fit_rows`, `eval_rows`, `degree_eval` and `fallback_probability`. Inner files additionally bind their validation `y_eval`. Outer-test labels stay outside feature building and model selection.

## Train and evaluate

The suite evaluates both learning rates at every five-epoch checkpoint through the selected budget, averages three inner raw Macro-F1 scores and breaks ties by lower learning rate then earlier epoch. It fits AB/ABR/Full with final seeds 2026111601/02/03. Configuration JSON files document the settings; the executable uses the explicit `--budget` option.

```sh
python scripts/train_classifier.py --suite work/scopes --task E --out work/E_budget80 --budget 80 --device cuda
python scripts/train_classifier.py --suite work/scopes --task E --out work/E_budget160 --budget 160 --device cuda
```

For the tested E fold-0 Full recipe:

```sh
python scripts/train_classifier.py --input work/scopes/E_o0.npz --out work/E_o0_Full --arm Full --lr 0.001 --epochs 140 --seed 2026111601 --device cuda
```

After each seed's five outer predictions exist, supply them together to evaluation:

```sh
python scripts/evaluate_oof.py --truth work/E/E__outer_truth.npz --predictions work/E_budget80/E_o0_Full/seed2026111601/prediction.npz work/E_budget80/E_o1_Full/seed2026111601/prediction.npz work/E_budget80/E_o2_Full/seed2026111601/prediction.npz work/E_budget80/E_o3_Full/seed2026111601/prediction.npz work/E_budget80/E_o4_Full/seed2026111601/prediction.npz --out work/E_full_seed1_metrics.json
```

The evaluation requires exactly one prediction per row and reports both raw and operational probabilities. Operational output uses the common fit-specific LR-AB probability for both-endpoint-zero evidence. Main scores are four-class pooled OOF metrics per seed, then a three-seed summary; per-fold scores and seed standard deviations are separate summaries.

## Replay results and statistical comparisons

With the hash-identified retained inputs:

```sh
python scripts/method.py check-scores AUTHORIZED_PREDICTIONS
python scripts/verify_reported_results.py --stat-assets AUTHORIZED_BLOBS --out work/intervals54.csv
```

The first command checks 72 score records from 12 pooled prediction files. The second replays the 54 original comparisons specified by `manifests/original54_replay.json`. Statistical blobs use the filename `<sha256>.npz`; the manifest binds dependency arrays, probability keys, seed weights, comparison signs and family sizes. Obtain these retained inputs under the access conditions in DATA_ACCESS.

For new output files, `scripts/statistical_analysis.py` takes a dependency NPZ (`rows`, `y`, integer-coded `fact`, `source`, `target`, `pair` groups) and a JSON list of signed prediction terms. Each term contains `key`, `sign`, `file` and optional `probability_key`/`seed`. Prediction files contain `rows` and a four-column probability array; paths are resolved relative to the term file.

```sh
python scripts/statistical_analysis.py --dependency-data DEPENDENCIES.npz --terms TERMS.json --family-size 12 --out work/contrast.json
python scripts/generate_tables.py --input results/main_seed_summary.csv --out work/tables
```

The 78 reported comparisons comprise the original 54 (four families of 12 and one of 6) plus two 12-comparison budget families. Each family retains its own Bonferroni adjustment. Intervals condition on saved predictions and the declared dependence assumptions; two-way clustering combines source and target components minus the pair component. The 160-minus-80 contrasts retain same-query and same-seed pairing. Aggregate interval files retain zero/negative-variance and low-support statuses where applicable. Protocol/source hashes in the family and implementation registries identify the original reference versions; the release manifest identifies the current public files.

## Tested scope and numerical conditions

Data parsing/identity restoration, feature/classifier components, result replay, statistics and plotting entries have been exercised. In a separate environment, E outer fold 0, seed 2026111601, Full was rebuilt from RDF with declared retained role/split metadata, a scoped projection and name vectors. The 650 prediction labels agreed; raw/operational probability differences were `2.1219253540039062e-5` and `3.306195219654473e-5`, with equal Macro-F1 (`0.9324948413357547`).

OpenBLAS versus MKL produced a maximum standardized-Q63 difference of `8.392333984375e-5`; the same-input MKL control was exact. B7 differed by `1.0728836059570312e-6`, and refitted common-LR probabilities by `5.149706329810044e-5`. The Q63/common-LR intermediate differences exceed the earlier `2e-5`/`1e-5` checks, while the final E prediction differences meet `1e-4` and preserve labels. The measurements are retained in [reproduction_measurements.json](../verification/reproduction_measurements.json).

The independent rerun covers this specified E scope and retained-input result checks. A complete independent four-task run and retraining of every comparison family remain outside that tested scope. Timing files describe their stated component/cache conditions; they are not whole-deployment latency measurements.
