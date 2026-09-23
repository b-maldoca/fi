---
name: doc-lockstep
description: Enforces strict lockstep synchronization between code changes and README.md, docs/prd_early_retirement_calc.md, and docs/design_doc_early_retirement.md in the Zurich Early Retirement Simulator (fi). Use whenever adding or modifying simulation logic, data sources, SimConfig parameters, tax rules, or Streamlit UI controls.
---

# Documentation Lockstep Skill (`doc-lockstep`)

## When to Use
Invoke and follow this checklist before finishing **any** task in the `fi` repository that changes:
- `SimConfig` fields, defaults, or `__post_init__` validation rules (`src/simulation_engine.py`)
- Data sources, CSV files in `data/`, `scripts/fetch_source_data.py`, or `scripts/build_historic_returns.py` (`src/historic_returns.py`)
- Tax brackets, indexation rules, or AHV / Pillar 2 / Pillar 3a logic (`src/tax_engine.py`, `src/simulation_engine.py`)
- Streamlit sidebar controls, defaults, charts, or summary tables (`app.py`)

## Lockstep Synchronization Checklist

1. **[`README.md`](../../README.md)**:
   - Update **Key Features** if user-facing behavior, asset sleeves, or tax/spending strategies changed.
   - Update **Data Pipeline & Provenance** if any dataset in `data/` or generator logic in `scripts/` changed.
   - Update **Project Structure** if files or scripts were added, renamed, or removed.

2. **[`docs/prd_early_retirement_calc.md`](../../docs/prd_early_retirement_calc.md)**:
   - Update **§3 Scope & Phasing** if high-level capabilities changed.
   - Update **§4.1 Data Inputs** for any new/modified sidebar sliders, number inputs, selectboxes, or defaults.
   - Update **§4.2 Emulation Engine & Calculation Logic** for any changes to tax formulas, pension rules, or return/inflation generators.
   - Update **§4.3 Outputs & Visualizations** if metrics, charts, or tables changed.

3. **[`docs/design_doc_early_retirement.md`](../../docs/design_doc_early_retirement.md)**:
   - Update **§3.1 User State & Configuration** validation rules and parameter descriptions.
   - Update **§3.2 Tax & Pension Engine** and **§3.3 Simulation Engine (Monthly Tick Execution Order)** if core math changed.
   - Update **§3.4 Return Generators & Data Provenance** if `scripts/build_historic_returns.py` or `data/` changed.
   - Update **§4 Data Models** (`@dataclass class SimConfig`) so the code block in the Design Doc matches `src/simulation_engine.py` field-for-field.
   - Update **§5 UI Layout (Streamlit)** so the sidebar section count and widget list match `app.py`.
