# Actual validation

The candidate was copied to an independent directory and run in a new virtual environment with no system site packages. Exact installed PyTorch/NumPy distributions were physically copied and SHA-checked; remaining numerical/plotting dependencies were separately installed. A file-access guard rejected implicit access to original study/manuscript directories. All declared validation runs report zero forbidden reads.

Original54 interval replay ran using119 explicitly copied, SHA-verified authorized data blobs. `python scripts/verify_reported_results.py --stat-assets AUTHORIZED_BLOBS --out check.csv` runs all five original comparison families. These supplied blobs are not in the public ZIP.

The E external fold0 Full classifier used seed2026111601 and the actual160-budget inner-selected recipe: {"lr": 0.001, "epochs": 140, "mean_inner_macro": 0.869700488112325}. Rows: 650. Prediction agreement: 1.00000000; raw max difference: 2.121925354e-05; operational max difference: 3.30619522e-05; Macro-F1 difference: 0. See LIMITATIONS for the retained projector/name vectors, historical metadata and measured numeric variation.

Quick Start, Verify Results, Rebuild Dataset, Rebuild Features, Reproduce One Fold and Full Reproduction entries are documented in README. A complete runnable entry is distinct from an actually completed all-experiment independent rerun.
