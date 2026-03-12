import shutil, logging, threading
from pathlib import Path
from PyQt6.QtCore    import QObject, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QApplication
from core.classifier       import FileClassifier
from core.intention_sensor import IntentionSensor
from core.file_watcher     import FileWatcher, FileEvent
from ui.notification_widget import NotificationWidget
from ui.tray_app            import TrayApp

logger = logging.getLogger(__name__)

# This is the "Hearing" part we missed!
DEFAULT_DIRS = [Path.home()/"Downloads", Path.home()/"Desktop"]

class _Bridge(QObject):
    new_file = pyqtSignal(object)

class AppController(QObject): 
    def __init__(self, watch_dirs=None):
        super().__init__() 
        # Ensure we are actually watching the right folders
        self.watch_dirs = [d for d in (watch_dirs or DEFAULT_DIRS) if d.exists()]
        self.classifier = FileClassifier()
        self._sensor = IntentionSensor()
        
        # Pass the actual directories to the watcher
        self._watcher = FileWatcher(self.watch_dirs, self._on_file_bg, self._sensor)
        
        self._bridge = _Bridge()
        self._bridge.new_file.connect(self._on_file_main)
        self._open = {}

    def start(self, app: QApplication):
        self._tray = TrayApp(self)
        self._tray.show()
        self._sensor.start()
        self._watcher.start()
        logger.info(f"System Active. Watching: {[d.name for d in self.watch_dirs]}")

    def shutdown(self):
        self._watcher.stop()
        self._sensor.stop()

    def _on_file_bg(self, event: FileEvent):
        self._bridge.new_file.emit(event)

    @pyqtSlot(object)
    def _on_file_main(self, event: FileEvent):
        fp = str(event.filepath)
        if fp in self._open: return

        print(f"DEBUG: AI Sensed App: {event.source_app} | URL Hint: {event.source_url}")

        # 1. Prediction
        feat = FileClassifier.build_feature_string(event.filename, event.extension, event.source_app, event.source_url, event.hour)
        top_matches = self.classifier.get_top_classes(feat, 3)
        
        folder, conf = (top_matches[0][0], top_matches[0][1]) if top_matches else ("Uncategorized", 0.0)

        # 2. Package context so the AI can learn from it later
        ctx = {'app': event.source_app, 'url': event.source_url, 'hour': event.hour}
        
        # 3. Show Widget
        w = NotificationWidget(fp, folder, conf, top_matches, ctx)
        w.accepted.connect(self._accepted)
        w.rejected.connect(self._rejected)
        w.dismissed.connect(self._dismissed)
        
        w.setWindowFlags(w.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        w.show_animated()
        w.raise_()
        w.activateWindow()
        self._open[fp] = w

    @pyqtSlot(str, str, str, str, int)
    def _accepted(self, filepath, folder, app, url, hour):
        self._open.pop(filepath, None)
        src = Path(filepath); dst_dir = Path(folder)
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name; i = 1
            while dst.exists(): dst = dst_dir / f"{src.stem}_{i}{src.suffix}"; i+=1
            
            shutil.move(str(src), str(dst))
            # Learning happens here
            feat = FileClassifier.build_feature_string(src.name, src.suffix.lstrip("."), app, url, hour)
            threading.Thread(target=self.classifier.train, args=(feat, folder), daemon=True).start()
            self._tray.notify("Success ✓", f"Moved to {dst_dir.name}")
        except Exception as e: logger.error(f"Move error: {e}")

    @pyqtSlot(str, str, str, str, int)
    def _rejected(self, filepath, folder, app, url, hour):
        self._open.pop(filepath, None)
        src = Path(filepath); dst_dir = Path(folder)
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name; i = 1
            while dst.exists(): dst = dst_dir / f"{src.stem}_{i}{src.suffix}"; i+=1
            
            shutil.move(str(src), str(dst))
            # Learning happens here
            feat = FileClassifier.build_feature_string(src.name, src.suffix.lstrip("."), app, url, hour)
            threading.Thread(target=self.classifier.train, args=(feat, folder), daemon=True).start()
            self._tray.notify("Learned 📚", f"Now pointing to {dst_dir.name}")
        except Exception as e: logger.error(f"Training error: {e}")

    @pyqtSlot(str)
    def _dismissed(self, filepath):
        self._open.pop(filepath, None)