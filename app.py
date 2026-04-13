import shutil, logging, threading
from pathlib import Path
from PyQt6.QtCore    import QObject, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QApplication
from core.classifier       import FileClassifier, AUTO_MOVE_THRESHOLD
from core.intention_sensor import IntentionSensor
from core.file_watcher     import FileWatcher, FileEvent
from ui.notification_widget import NotificationWidget
from ui.tray_app            import TrayApp
 
logger = logging.getLogger(__name__)
 
DEFAULT_DIRS = [Path.home()/"Downloads", Path.home()/"Desktop"]
 
class _Bridge(QObject):
    new_file = pyqtSignal(object)
 
class AppController(QObject):
    def __init__(self, watch_dirs=None, seed_folders=None):
        super().__init__()
        self.watch_dirs = [d for d in (watch_dirs or DEFAULT_DIRS) if d.exists()]
        self.classifier = FileClassifier()
        self._sensor = IntentionSensor()
        self._watcher = FileWatcher(self.watch_dirs, self._on_file_bg, self._sensor)
        self._bridge = _Bridge()
        self._bridge.new_file.connect(self._on_file_main)
        self._open = {}
        self.threshold = AUTO_MOVE_THRESHOLD  # expose for SettingsDialog slider
        self.auto_move_threshold = AUTO_MOVE_THRESHOLD
 
        # Seed the classifier with known folders so it has a prior immediately
        folders_to_seed = seed_folders or []
        # Also auto-seed from any known training folders already in the DB
        existing = self.classifier.get_all_folders()
        all_seed = list({str(f) for f in folders_to_seed} | set(existing))
        if folders_to_seed:
            logger.info("Seeding classifier with %d folders", len(folders_to_seed))
            threading.Thread(
                target=self.classifier.seed_folders,
                args=(folders_to_seed,),
                daemon=True
            ).start()
 
    def start(self, app: QApplication):
        self._tray = TrayApp(self)
        self._tray.show()
        self._sensor.start()
        self._watcher.start()
        logger.info(f"System Active. Watching: {[d.name for d in self.watch_dirs]}")
 
    def shutdown(self):
        self._watcher.stop()
        self._sensor.stop()
 
    def add_watch_directory(self, path):
        p = Path(path)
        if p.exists() and p not in self.watch_dirs:
            self.watch_dirs.append(p)
            self._watcher.add_directory(p)
 
    def _on_file_bg(self, event: FileEvent):
        self._bridge.new_file.emit(event)
 
    @pyqtSlot(object)
    def _on_file_main(self, event: FileEvent):
        fp = str(event.filepath)
        if fp in self._open:
            return
 
        logger.debug(f"New file: {event.filename} | App: {event.source_app} | URL: {event.source_url}")
 
        feat = FileClassifier.build_feature_string(
            event.filename, event.extension,
            event.source_app, event.source_url, event.hour
        )
        top_matches = self.classifier.get_top_classes(feat, 3)
        folder, conf = (top_matches[0][0], top_matches[0][1]) if top_matches else ("", 0.0)
 
        ctx = {'app': event.source_app, 'url': event.source_url, 'hour': event.hour}
 
        # ── Auto-move when confidence is very high ──────────────────────────
        if folder and conf >= self.auto_move_threshold:
            logger.info(f"Auto-moving {event.filename} → {folder} (conf={conf:.0%})")
            self._do_move(event.filepath, folder, event.source_app, event.source_url, event.hour, auto=True)
            return
 
        # ── Otherwise show the notification widget ──────────────────────────
        w = NotificationWidget(fp, folder, conf, top_matches, ctx)
        w.accepted.connect(self._accepted)
        w.rejected.connect(self._rejected)
        w.dismissed.connect(self._dismissed)
        w.setWindowFlags(w.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        w.show_animated()
        w.raise_()
        w.activateWindow()
        self._open[fp] = w
 
    # ── shared move logic ───────────────────────────────────────────────────
    def _do_move(self, src: Path, dst_folder: str, app: str, url: str, hour: int, auto: bool = False):
        src = Path(src); dst_dir = Path(dst_folder)
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name; i = 1
            while dst.exists():
                dst = dst_dir / f"{src.stem}_{i}{src.suffix}"; i += 1
            shutil.move(str(src), str(dst))
            feat = FileClassifier.build_feature_string(src.name, src.suffix.lstrip("."), app, url, hour)
            threading.Thread(target=self.classifier.train, args=(feat, dst_folder), daemon=True).start()
            label = "Auto-moved 🤖" if auto else "Moved ✓"
            self._tray.notify(label, f"{src.name} → {dst_dir.name}")
        except Exception as e:
            logger.error(f"Move error: {e}")
 
    @pyqtSlot(str, str, str, str, int)
    def _accepted(self, filepath, folder, app, url, hour):
        self._open.pop(filepath, None)
        self._do_move(Path(filepath), folder, app, url, hour)
 
    @pyqtSlot(str, str, str, str, int)
    def _rejected(self, filepath, folder, app, url, hour):
        """User picked a different folder — train with the corrected label."""
        self._open.pop(filepath, None)
        src = Path(filepath); dst_dir = Path(folder)
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name; i = 1
            while dst.exists():
                dst = dst_dir / f"{src.stem}_{i}{src.suffix}"; i += 1
            shutil.move(str(src), str(dst))
            feat = FileClassifier.build_feature_string(src.name, src.suffix.lstrip("."), app, url, hour)
            threading.Thread(target=self.classifier.train, args=(feat, folder), daemon=True).start()
            self._tray.notify("Learned 📚", f"Now pointing to {dst_dir.name}")
        except Exception as e:
            logger.error(f"Training error: {e}")
 
    @pyqtSlot(str)
    def _dismissed(self, filepath):
        self._open.pop(filepath, None)