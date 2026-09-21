# Technical Decisions

## ADR-001: Preserve fixed FD001 contracts in typed project configuration

- Status: Accepted
- Context: The roadmap freezes dataset, target cap, minimum history, seed, and feature schema values.
- Decision: Expose these values through a small Pydantic configuration model and keep runtime commands separate from future pipeline implementations.
- Alternatives considered: Ad-hoc constants in each module, which risks contract drift; a full settings service, which is unnecessary for the local foundation.
- Consequences: Later modules have a single source for core constants and can validate configuration at startup.
- Evidence: `src/config.py`, `tests/test_smoke.py`
