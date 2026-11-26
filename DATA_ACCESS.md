# Data and model access

## Benchmark

The four standard pairs come from [Zenodo record 17091043](https://doi.org/10.5281/zenodo.17091043). The archive is `benchmark.zip`, SHA-256 `04cc6dd2a2f8d173ddcc19822428fa9f505e3855840f06b19072306233d240eb`. Individual RDF checksums appear in [data_hashes.csv](manifests/data_hashes.csv) and [dataset_sources.csv](manifests/dataset_sources.csv).

## Download and parse

```sh
python scripts/fetch_sources.py --out work/benchmark.zip --sha256 04cc6dd2a2f8d173ddcc19822428fa9f505e3855840f06b19072306233d240eb --url https://zenodo.org/api/records/17091043/files/benchmark.zip/content --extract
python scripts/audit.py --input work/benchmark/etim-eclass --task E --out work/audit_E
```

Use `--local PATH` instead of `--url URL` for an existing authorized archive. Extraction creates `work/benchmark`; adjust the dataset directory if the archive has another enclosing directory. `verify_dataset.py` checks the three RDF hashes before restoring the retained rows and splits.

## Models

The name encoder is [all-MiniLM-L6-v2, revision 1110a243](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41), using `onnx/model_quint8_avx2.onnx` and `tokenizer.json`. The optional text comparisons use [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) at revision `86b5e0934494bd15c9632b12f734a8a67f723594` and the [TaSeR author archive](https://doi.org/10.6084/m9.figshare.21750338.v1), including `TaSeR/pytorch_model.bin`.

Download model files from these author sources. [model_sources.csv](manifests/model_sources.csv) records checkpoint hashes and the corresponding source terms. `scripts/encode_names.py` encodes an ordered JSON list of names with explicitly supplied ONNX/tokenizer files. The guide describes how its outputs map to source/target vector arrays.

## Reproduction inputs

The split CSVs contain reference-cell indices, row identifiers, dependency-group hashes and fold assignments. Join them against the matching RDF snapshot to obtain names and labels. Historical role/Guard and cross-source screening decisions are retained metadata.

Name vectors, fit-specific projectors, individual prediction files and statistical dependency arrays are additional inputs for exact retained-asset replay. Their schemas and hash conventions are in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Where rights permit, availability can be discussed through the repository issue tracker; retained input access is subject to the source terms. Newly rebuilt features can be evaluated with their own provenance and numerical environment.

## License

The repository provides project code, aggregate results and split metadata under MIT. Taxonomy RDF/text/reference mappings and model weights are obtained separately under their applicable benchmark, ETIM/ECLASS/GPC/UNSPSC and model-source terms. Check these terms before downloading or redistributing the corresponding assets; the code license covers project-owned materials.
