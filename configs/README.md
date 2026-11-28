# Configuration records and executable settings

`method.json` records the feature/encoder/fallback recipes. The `method.py` CLI reads its `label_order` to check the four-class convention. Feature builders otherwise use their retained implementation constants and explicit arguments; `q63_kernel.py` holds the fixed beta values.

`budget80.json` and `budget160.json` document the two reported MLP families. They are not auto-loaded training configurations. Select a suite with `train_classifier.py --budget 80` or `--budget 160`; the default suite ceiling remains 160. `--epochs` instead controls a single `--input` run. No CLI option implicitly loads one of these JSON files.

Statistical replay is different: `verify_reported_results.py` reads `manifests/original54_replay.json`, and `statistical_analysis.py --terms FILE.json` reads the supplied signed prediction terms. Those are executable input specifications.

Numeric settings, seeds, tolerances, prefix checks and continuation safeguards are preserved from the original recipes. The 80/160 families use the same 146-coordinate layout and common zero-evidence fallback.
