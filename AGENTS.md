# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

Crystal Mancer is a Python-based AI catalyst discovery engine (Phase 1: knowledge extraction pipeline). Single-service Python application — no containers, databases, or external services needed for core development.

### Setup Gotcha

The `pyproject.toml` includes a `[tool.setuptools.packages.find]` section that restricts package discovery to `crystalmancer*`. Without this, `setuptools` fails with a flat-layout error because it discovers `viewer/`, `notebooks/`, and `autoresearch/` as top-level packages.

### Running the application

- **Dry run (no API calls):** `python3 -m crystalmancer --dry-run -v`
- **CLI help:** `python3 -m crystalmancer --help`
- Alternatively, use `crystalmancer` directly if `~/.local/bin` is on PATH.

### Testing

- **Run all tests:** `python3 -m pytest tests/ -v`
- All 43 tests use mocks; no network or external services required.

### Linting

No linting tools (ruff, flake8, mypy, etc.) are configured in this project.

### External services

- **Neo4j, OpenRouter API, Materials Project API** are optional and only needed for full pipeline runs with real data. The dry-run mode and test suite work without any external services or API keys.
- A local NetworkX+SQLite fallback exists for the knowledge graph, so Neo4j is never required for basic development.
