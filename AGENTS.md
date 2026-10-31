# Portable model work

- Run the synthetic tests before publishing changes. Do not train on real data in CI.
- Keep original data, identifiers, coordinates, per-row predictions and trained weights local. Never print rows. Detailed CSV exports must go through `oibc_portable.local_details.write_csv` into `.local-only/details/`.
- Keep the target-day/end-of-hour convention and the previous-day17:00KST forecast origin explicit. Labels used for fitting must be available at that origin. Selection uses a full-day embargo and the final nonnegative output metric.
- Preserve completed artifacts; resume only under the same input, implementation and configuration contract. Do not replace existing models or source data.
- RAW6 is262 columns. The no-weather control retains only metadata5, calendar4 and solar6. Do not call direct-forecast-column removal a complete meteorological-information removal.
- Model gain, split count and CatBoost PredictionValuesChange are usage diagnostics, not causal effects or held-out accuracy contributions.
- Publish only the allowlisted source, documentation, configuration and synthetic tests using `tools/check_release.py`.
