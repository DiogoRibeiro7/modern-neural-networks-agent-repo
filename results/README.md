# Raw experiment records

Every file here is one seeded run serialized with the schema in
`src/modern_nn_lab/experiments/records.py`. Records are committed so that every table
and figure in `reports/` can be regenerated without rerunning training.

Layout: `results/<track>/<architecture>__<variant>__<dataset>__seed<k>.json`.

Validate with `python scripts/validate_results.py`, summarize with `modern-nn summarize`.
