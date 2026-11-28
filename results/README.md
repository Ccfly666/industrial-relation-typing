# Results and manuscript locations

This index follows the refined V5 manuscript. Printed table numbers can change; existing `table_id` values in `paper_tables.csv` are stable research keys.

| V5 table | Stable key | Files | Experimental scope |
| --- | --- | --- | --- |
| 5 | `main` | [budget_main_results.csv](budget_main_results.csv), [main_seed_summary.csv](main_seed_summary.csv) | Common MLP; 80 and 160 ceilings |
| 6 | `AB` | [paper_tables.csv](paper_tables.csv) | Separate Tanh MLP and fixed LR |
| 7 | `R1` | [paper_tables.csv](paper_tables.csv), [r1_six_intervals.csv](r1_six_intervals.csv) | Fixed LR; E/C/U |
| 8 | `MLP` | [paper_tables.csv](paper_tables.csv), [mechanism_intervals.csv](mechanism_intervals.csv) | Original 80-epoch MLP mechanisms |
| 9 | `direct` | [paper_tables.csv](paper_tables.csv), [mechanism_intervals.csv](mechanism_intervals.csv) | Matched Top32 events; three seeds |
| 10 | `baselines` | [paper_tables.csv](paper_tables.csv) | Overall models with respective training recipes |
| 11 | `strong` | [paper_tables.csv](paper_tables.csv) | Supervised BERT/TaSeR and fixed-LR readouts |
| E.1 | `supplement:seeds` | [budget_summary.csv](budget_summary.csv), [paper_tables.csv](paper_tables.csv) | Three seeds; both common ceilings |
| E.2 | `supplement:raw` | [budget_main_results.csv](budget_main_results.csv), [paper_tables.csv](paper_tables.csv) | Raw and operational; both ceilings |
| E.3 | `supplement:mainCI` | [main_intervals.csv](main_intervals.csv), [budget_statistical_intervals.csv](budget_statistical_intervals.csv) | 80/160 stage families separately adjusted |
| E.4 | `supplement:R1CI` | [r1_six_intervals.csv](r1_six_intervals.csv) | Six fixed-LR comparisons |
| E.5 | `supplement:LRCI` | [mechanism_intervals.csv](mechanism_intervals.csv), [paper_tables.csv](paper_tables.csv) | Twelve LR mechanism comparisons |
| E.6 | `supplement:MLPCI` | [mechanism_intervals.csv](mechanism_intervals.csv), [paper_tables.csv](paper_tables.csv) | Twelve original MLP mechanism comparisons |
| E.7 | `supplement:AGGCI` | [mechanism_intervals.csv](mechanism_intervals.csv), [paper_tables.csv](paper_tables.csv) | Twelve matched-event aggregation comparisons |
| E.8 | `concentration` | [cluster_concentration_localization.csv](cluster_concentration_localization.csv) | Maximum paired-IF concentration; 36 mechanism comparisons |
| E.9 | `classes` | [paper_tables.csv](paper_tables.csv) | Full class F1; full precision/recall/support in Online Resource 2 |
| E.10 | `coverage` | [paper_tables.csv](paper_tables.csv) | Original 80-epoch coverage-specific gains |
| E.12 | `sensitivity` | [cluster_sensitivity_results.csv](cluster_sensitivity_results.csv) | Fourteen descriptive endpoint deletions |
| E.13 | `budget_change` | [budget_statistical_intervals.csv](budget_statistical_intervals.csv) | Twelve paired 160-minus-80 gain changes |
| E.14 | `timing` | [inference_timing_summary.csv](inference_timing_summary.csv) | Cached ABR/name vectors; ms/query |
| D.2 | `Q28` | [compact_q28_displayed.csv](compact_q28_displayed.csv) | E/C/U LR; displayed precision only |

Main Tables 1–3 define the B7/Q63 layouts and dataset supports; Table 4 is the experimental-settings index. They are protocol descriptions rather than performance comparisons. SI Tables A.1–B.8 contain dataset/configuration/parameter details. SI Tables C.1–C.6 and D.1–D.2 contain the auxiliary representation, aggregation, LR and coordinate-recovery results. Full class metrics, coverage supports (E.11), the original auxiliary table rows and all 78 interval rows are supplied with the paper in **Online Resource 2 (`ESM_2.zip`)**, whose `index.csv` specifies every file's scope and units. The supplement uses 34 numbered tables; the front question/evidence guide is unnumbered.

`paper_tables.csv` contains 601 preserved cells with the original stable identifiers. For example, `supplement:extraSeeds` contains 72 original mechanism seed scores distributed with Online Resource 2, rather than a new printed table. Its `main` key records the original 80-epoch table; use `budget_main_results.csv` and `budget_summary.csv` for both ceilings. The historical `table_E9_maximum` field in concentration files refers to the current **E.8** concentration table, not E.9 class F1. `compact_q28_displayed.csv` transcribes the retained SI D.2 display; rounded score subtraction need not equal its gain computed from unrounded scores.

## Units and identities

| File/fields | Meaning |
| --- | --- |
| `main_seed_summary`: `mean`, `sd`, `values`, `mean_f1` | Fractions; `sd` is sample seed SD, not a confidence interval |
| `budget_main_results`, `budget_summary` | Fraction scores; 80/160 denote candidate ceilings; `values` follows the three head seeds |
| `main_intervals`: `delta`, `lo`, `hi`, `se`, `halfwidth` | Fraction differences; multiply by 100 for pp; `variance` is squared fraction units |
| `mechanism_intervals`, `budget_statistical_intervals`, `r1_six_intervals` | Same fraction/variance convention; `family_size` binds its own adjustment |
| `paper_tables`: `original_value`, `displayed_value`, `metric` | Units are explicit in `metric`; score/SD `%` and gain `pp` differ from fraction-valued aggregates |
| `experiment_family_registry`: `family`, `size`, `tasks`, `definition`, `training_budget`, `adjustment` | Frozen comparison family identity, number of comparisons, scope and multiplicity rule |
| `experiment_family_registry`: `source_config_sha256`, `recipe_file` | Historical configuration identity and its recipe file; not the hash of a rewritten current document |
| `implementation_identity_registry`: `source_name`, `source_sha256` | Original development source identities, not repository-relative download paths |
| `cluster_concentration_localization` | `cluster_N`, `N`, supports are counts; shares are fractions; `cluster_index` is encoded, not a URI |
| `cluster_sensitivity_results`: `*_pp`, `*_support`, `seed_*` | Percentage-point effects, class counts and JSON seed arrays; fixed-prediction descriptive deletions |
| `inference_timing_summary` | `*_ms` milliseconds per query; batch elapsed time is divided by batch size before percentiles |
| `memory_summary` | Bytes; sampled CPU RSS is not a true peak; GPU allocation/reservation includes the stated co-resident models |
| `runtime_parameter_counts` | Integer scalar-parameter counts; registered and effective path are distinct |
| `training_cost_by_task_arm` | Wall-clock seconds and update counts, with new/continued/reused work distinguished |
| `training_fold_selection`, `training_budget_diagnostics` | Actual selected recipes, checkpoint/update diagnostics and provenance; JSON-valued cells remain encoded arrays/objects |

The [current code hashes](../manifests/code_hashes.csv) and [release-file manifest](../manifests/release_file_manifest.csv) identify distributed files. Historical hashes in the research registries remain unchanged.
