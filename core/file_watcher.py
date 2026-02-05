import time, logging, mimetypes, threading, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

logger = logging.getLogger(__name__)
IGNORED_EXT  = {".tmp",".part",".crdownload",".download",".lock",".lnk",".ds_store","~"}
MIN_SIZE     = 10
DEBOUNCE_SEC = 1.5

@dataclass
class FileEvent:
    filepath:   Path
    filename:   str
    extension:  str
    mime_type:  str
    size_bytes: int
    source_app: str
    source_url: str
    hour:       int
    timestamp:  float = field(default_factory=time.time)

class _Handler(FileSystemEventHandler):
    def __init__(self, callback, sensor):
        super().__init__()
        self._cb = callback; self._sensor = sensor
        self._pending = {}; self._lock = threading.Lock()

    def on_created(self, event: FileCreatedEvent):
        if event.is_directory: return
        path = Path(event.src_path)
        if path.suffix.lower() in IGNORED_EXT or path.name.startswith("."): return
        key = str(path)
        with self._lock:
            if key in self._pending: self._pending[key].cancel()
            t = threading.Timer(DEBOUNCE_SEC, self._process, args=(path,))
            self._pending[key] = t; t.start()

    def _process(self, path: Path):
        with self._lock: self._pending.pop(str(path), None)
        if not path.exists(): return
        try: stat = path.stat()
        except OSError: return
        if stat.st_size < MIN_SIZE: return
        mime, _ = mimetypes.guess_type(str(path))
        ctx = self._sensor.get_dominant_context(seconds=5)
        self._cb(FileEvent(
            filepath=path, filename=path.name,
            extension=path.suffix.lstrip(".").lower(),
            mime_type=mime or "application/octet-stream",
            size_bytes=stat.st_size,
            source_app=ctx.app_name, source_url=ctx.source_url,
            hour=datetime.datetime.now().hour,
        ))

class FileWatcher:
    def __init__(self, watch_dirs, callback, sensor, recursive=False):
        self._dirs     = [Path(d).expanduser().resolve() for d in watch_dirs]
        self._observer = Observer()
        self._handler  = _Handler(callback, sensor)
        self._recursive = recursive

    def start(self):
        for d in self._dirs:
            if d.exists(): self._observer.schedule(self._handler, str(d), recursive=self._recursive)
            else: logger.warning("Watch dir does not exist: %s", d)
        self._observer.start()

    def stop(self):
        self._observer.stop(); self._observer.join()

    def add_directory(self, path):
        p = Path(path).expanduser().resolve()
        if p.exists():
            self._observer.schedule(self._handler, str(p), recursive=self._recursive)
            self._dirs.append(p)
