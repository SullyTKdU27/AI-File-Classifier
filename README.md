# AI File Classification System
**COMP1682 Final Year Project — Mohammed Suleman Faisal (001312934)**
Supervisor: Mobolaji Orisatoki | University of Greenwich

---

## Quick Start
```bash
pip install PyQt6 watchdog scikit-learn numpy psutil pytest

python demo.py --simulate    # verify ML works (no GUI needed)
python demo.py               # interactive training/prediction
python main.py               # full app (system tray + notifications)
python -m pytest tests/ -v   # run test suite
```

## Architecture
**Layer 1 — Ingestion:** `FileWatcher` (Watchdog, OS-native events) + `IntentionSensor` (foreground window polling, 5s rolling buffer)

**Layer 2 — ML Engine:** Multinomial Naive Bayes + TF-IDF. Adaptive threshold: `max(0.40, 1/n_classes + 0.15)`. SQLite persistence.

**Layer 3 — HITL UI:** PyQt6 notification card (bottom-right, 12s auto-dismiss). Accept → move + train. Reject → folder picker + train.

## Key Design Decisions
- **Adaptive threshold** solves cold-start suppression (fixed 0.65 silenced correct predictions with 4+ classes)
- **Full re-fit** on every training event (not partial_fit) — TF-IDF vocabulary must be rebuilt consistently; <100ms for realistic set sizes
- **Dominant context** (most frequent app in 5s buffer) is more robust than latest-snapshot for brief focus changes
- **Position boost** — first filename token repeated 3x for higher TF-IDF weight

## Known Limitations
1. Cold start: needs ≥2 folder classes before predicting. Seed with `demo.py`.
2. Scales to ~500 training examples before full re-fit becomes slow.
3. URL extraction from browser titles relies on keyword matching, not actual URL access.
