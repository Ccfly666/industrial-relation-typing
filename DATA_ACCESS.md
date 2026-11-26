# Data and model access

## Source benchmark

The source benchmark is identified by [DOI 10.5281/zenodo.17091043](https://doi.org/10.5281/zenodo.17091043). The published member is `benchmark.zip`; the recorded SHA-256 is `04cc6dd2a2f8d173ddcc19822428fa9f505e3855840f06b19072306233d240eb`. The download and local-copy paths both validate that checksum. File-level RDF identities are in `manifests/data_hashes.csv` and `manifests/dataset_sources.csv`.

Use `scripts/fetch_sources.py --help` for acquisition and `scripts/audit.py --help` for parsing. Obtain authorization appropriate to the benchmark and the upstream ETIM, ECLASS, GPC and UNSPSC material. Public access and checksum verification do not establish unrestricted redistribution rights. This repository neither grants those rights nor republishes the original RDF, class descriptions, concept names, or reference mappings.

## Retained roles and dependencies

`manifests/*_retained_roles_splits.csv` contains reference-cell indices, formal row identifiers, role decisions, dependency-group hashes, and outer/inner assignments. It contains no taxonomy text or reference relation labels. These are the declared historical role/Guard decisions, not newly rediscovered data curation. They are included as project-produced reproducibility metadata. Join them only against a matching authorized source snapshot with `scripts/verify_dataset.py`.

## Text encoders and retained inputs

Model identifiers/revisions are recorded in `configs/verified_recipe.json` and the implementation registries. Obtain encoder weights from their authors under their own terms. `scripts/encode_names.py` is the name-vector construction entry. Fixed quantized batching/order matters for numerical replay.

Embeddings, trained projectors, classifier checkpoints, per-row predictions, and private statistical/acceptance inputs are not included. Their hashes identify expected assets but do not make them downloadable here. For strict retained-input replay, request authorized access through the repository issue tracker; availability is not guaranteed and restricted assets cannot be shared merely because a request is made. Alternatively, rebuild the allowed components and report the new provenance and numerical differences.

## Included results and license scope

The aggregate CSV results, split/dependency metadata, implementation code, and project documentation are released by the project rights holder. The MIT license covers project-owned materials to the extent of those rights. It does not relicense benchmark contents, model parameters, or third-party software. No manuscript, private review record, individual prediction file, or private validation input is distributed here.
