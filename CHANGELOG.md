# Changelog

All notable changes to this project will be documented in this file.

The format follows Keep a Changelog and the project versioning follows Semantic Versioning.

## [0.2.0] - 2026-03-10

### Added
- Async Telegram notification runtime with event-loop-managed startup and shutdown.
- PostgreSQL migration path, runtime database configuration, and optional Docker PostgreSQL profile.
- React real-time monitoring dashboard with authenticated WebSocket streaming.
- Backtest/live scoring parity adapter and parity tests.
- Backtest live-parity enforcement hook for strategy wrappers.
- Shadow-mode model A/B testing framework with JSONL comparison logs and same-module variant injection.
- Custom API usage guide and project semantic version source files.
- Decision variant profile file for baseline/candidate experiments.

### Changed
- API metrics endpoints now compute from the shared trade loader instead of SQLite-only SQL queries.
- Decision pipeline can run an optional non-trading shadow model comparison path.
