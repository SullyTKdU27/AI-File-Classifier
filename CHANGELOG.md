# CHANGELOG

## [v0.4.0] March 2026 — Evaluation & Hardening
### Fixed (Critical)
- **Adaptive threshold**: replaced fixed 0.65 with `max(0.40, 1/n_classes + 0.15)`. Fixed cold-start suppression where 3/4 correct predictions were silenced with 4 classes.
- Removed stale `CONFIDENCE_THRESHOLD` import from app.py (would crash on startup).
- Added `token_pattern=None` to TfidfVectorizer to suppress sklearn warning.
- `FileClassifier` now accepts `confidence_threshold` constructor arg for testing.

### Added
- Full 15-test suite in `tests/test_classifier.py`
- `README.md` and `CHANGELOG.md`
- `demo.py --simulate` for 4-week rejection rate simulation

## [v0.3.0] February 2026 — HITL Interface
### Added
- PyQt6 notification widget with confidence bar, accept/reject/dismiss
- System tray icon with Stats and Settings dialogs
- Qt signal bridge (pyqtSignal) for thread-safe background→UI marshalling

### Fixed
- Cross-thread GUI corruption: Watchdog callbacks now bridge to Qt main thread via pyqtSignal before any UI code executes.

## [v0.2.0] January 2026 — ML Engine
### Added
- Multinomial Naive Bayes + TF-IDF with position-boosting tokeniser
- Context signal repetition (app/URL repeated 3×)
- SQLite persistence for model weights and training history

### Changed (from rejected approach)
- **Replaced Pipeline+partial_fit with standalone vectoriser rebuilt on each training event.** Pipeline does not expose partial_fit on TfidfVectorizer — vocabulary was frozen after first fit, causing silent accuracy degradation as new tokens were ignored.

## [v0.1.0] December 2025 — Proof of Concept
### Added
- Watchdog observer with 1.5s debounce and extension filtering
- IntentionSensor with 5-second rolling buffer and dominant-context logic

### Rejected approach
- **Polling-based file detection** (checking directory every 2s): introduced up to 2s latency and continuous CPU overhead. Replaced with Watchdog OS-native events.
