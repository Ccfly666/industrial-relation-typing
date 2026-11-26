# Code and experiment provenance

The implementation combines supervised low-rank text projection, catalogue/name features, typed relation-neighbourhood profiles and query-conditioned summaries. [code_origins.csv](manifests/code_origins.csv) identifies the original research kernels and their reference hashes. Source names in that file and the implementation registry identify development versions rather than downloadable repository paths.

The acquisition and RDF parsing tools were reconstructed for the released snapshots. Row roles, dependency groups and splits are retained research metadata; E uses the corrected proof-group split. Their schemas and source checksums are supplied in `manifests/`.

The main experiments use the 80-epoch configuration. A separate 160-epoch analysis examines training-budget sensitivity after earlier baseline and early-stop observations. Table cells, family definitions and per-seed aggregates are available in `results/`.

[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) documents the tested environment, the E-fold reproduction and numerical conditions. [CHANGELOG.md](CHANGELOG.md) records software revisions; [release_file_manifest.csv](manifests/release_file_manifest.csv) identifies the current public files. Current Python file hashes are listed in [code_hashes.csv](manifests/code_hashes.csv).
