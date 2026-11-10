# Repository Rules & Conventions (`fi`)

## 1. Documentation Lockstep Requirement (MANDATORY)
Whenever you modify any simulation behavior, data source, return generator, `SimConfig` field, tax calculation, or Streamlit UI input/output in this repository, you **MUST** update all three documentation files in the same turn before concluding:
1. [`README.md`](README.md) — User-facing overview, data pipeline & provenance table, and project structure.
2. [`docs/prd_early_retirement_calc.md`](docs/prd_early_retirement_calc.md) — Product Requirement Document (functional inputs, calculation logic, return engines, and outputs).
3. [`docs/design_doc_early_retirement.md`](docs/design_doc_early_retirement.md) — Technical Design Document (`SimConfig` dataclass specification, validation rules, monthly tick execution order, return generators, and UI layout).

Never leave `README.md`, `docs/prd_early_retirement_calc.md`, or `docs/design_doc_early_retirement.md` out of sync with the code.

## 2. Intentional Scope Decisions (Working As Intended)
- **Asset Classes**: The simulator intentionally models 5 asset sleeves (`US Stocks`, `Non-US Stocks`, `CHF Cash`, `Gold`, `Bitcoin`). Do **not** add separate Swiss equities or bond sleeves unless explicitly instructed by the user.
- **Generated Files**: `src/historic_returns.py` (CHF) and `src/historic_returns_eur.py` (EUR / Austria) are generated offline by `scripts/build_historic_returns.py` (`--currency EUR` for the latter), with raw data fetched via `scripts/fetch_source_data.py` and `scripts/fetch_source_data_at.py`. Never hand-edit either generated file; modify `scripts/build_historic_returns.py` and re-run it.
