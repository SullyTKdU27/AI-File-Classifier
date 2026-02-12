import re, sys, time, logging, threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)
POLL_INTERVAL = 1.0
BUFFER_SIZE   = 10

@dataclass
class WindowSnapshot:
    title:      str
    app_name:   str
    source_url: str
    timestamp:  float = field(default_factory=time.time)

class IntentionSensor:
    def __init__(self):
        self._buffer: deque = deque(maxlen=BUFFER_SIZE)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._poll_fn = self._get_poll_fn()

    def _get_poll_fn(self):
        if sys.platform == "win32":   return self._poll_windows
        elif sys.platform == "darwin": return self._poll_macos
        else:                          return self._poll_linux

    def _poll_windows(self) -> WindowSnapshot:
        try:
            import win32gui, win32process, psutil
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:    app = psutil.Process(pid).name().replace(".exe","")
            except: app = ""
            return WindowSnapshot(title=title, app_name=app, source_url=self._url_from_title(title, app))
        except: return self._stub()

    def _poll_macos(self) -> WindowSnapshot:
        try:
            from AppKit import NSWorkspace
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            name = app.localizedName() if app else ""
            return WindowSnapshot(title=name, app_name=name, source_url=self._url_from_title(name, name))
        except: return self._stub()

    def _poll_linux(self) -> WindowSnapshot:
        try:
            import subprocess
            title = subprocess.run(["xdotool","getactivewindow","getwindowname"], capture_output=True, text=True, timeout=1).stdout.strip()
            app   = subprocess.run(["xdotool","getactivewindow","getwindowclassname"], capture_output=True, text=True, timeout=1).stdout.strip()
            return WindowSnapshot(title=title, app_name=app, source_url=self._url_from_title(title, app))
        except: return self._stub()

    @staticmethod
    def _url_from_title(title: str, app: str) -> str:
        s = (title + " " + app).lower()
        HINTS = {"scholar.google":"scholar.google.com","arxiv":"arxiv.org","youtube":"youtube.com",
                 "github":"github.com","stackoverflow":"stackoverflow.com","drive.google":"drive.google.com",
                 "moodle":"moodle","blackboard":"blackboard","overleaf":"overleaf.com",
                 "notion":"notion.so","figma":"figma.com","reddit":"reddit.com"}
        for kw, domain in HINTS.items():
            if kw in s: return domain
        m = re.search(r"https?://[^\s|]+", title)
        return m.group(0) if m else ""

    @staticmethod
    def _stub() -> WindowSnapshot:
        return WindowSnapshot(title="", app_name="", source_url="")

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="IntentionSensor")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=3)

    def _loop(self):
        while self._running:
            self._buffer.append(self._poll_fn())
            time.sleep(POLL_INTERVAL)

    def get_context(self) -> WindowSnapshot:
        return self._buffer[-1] if self._buffer else self._stub()

    def get_dominant_context(self, seconds: int = 5) -> WindowSnapshot:
        now    = time.time()
        recent = [s for s in self._buffer if now - s.timestamp <= seconds]
        if not recent: return self._stub()
        from collections import Counter
        counts = Counter(s.app_name for s in recent if s.app_name)
        if not counts: return recent[-1]
        top_app = counts.most_common(1)[0][0]
        for snap in reversed(recent):
            if snap.app_name == top_app: return snap
        return recent[-1]
