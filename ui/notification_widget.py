from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QProgressBar, QFrame)
from PyQt6.QtCore    import Qt, pyqtSignal, QPropertyAnimation, QPoint, QTimer
from pathlib import Path
import os
 
class NotificationWidget(QWidget):
    accepted  = pyqtSignal(str, str, str, str, int)
    rejected  = pyqtSignal(str, str, str, str, int)
    dismissed = pyqtSignal(str)
 
    STYLE = """
        QWidget#container {
            background-color: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 14px;
        }
        QLabel { color: #cdd6f4; font-family: 'Segoe UI', Arial; }
        QLabel#title { font-size: 12px; color: #a6adc8; }
        QLabel#filename { font-size: 13px; font-weight: bold; color: #cdd6f4; }
        QPushButton {
            border-radius: 6px; padding: 7px 10px;
            font-size: 12px; font-weight: bold; text-align: left;
            border: 1px solid transparent;
        }
        QPushButton#btn_top  { background-color: #313244; color: #cdd6f4; border-color: #45475a; }
        QPushButton#btn_top:hover  { background-color: #89b4fa; color: #1e1e2e; }
        QPushButton#btn_alt  { background-color: #1e1e2e; color: #a6adc8; border-color: #313244; }
        QPushButton#btn_alt:hover  { background-color: #313244; color: #cdd6f4; }
        QPushButton#btn_browse { background-color: #1e1e2e; color: #6c7086; border-color: #313244; }
        QPushButton#btn_browse:hover { color: #cdd6f4; background-color: #313244; }
        QProgressBar { height: 4px; background: #313244; border: none; border-radius: 2px; }
        QProgressBar::chunk { background: #89b4fa; border-radius: 2px; }
    """
 
    def __init__(self, filepath, folder, confidence, top_matches, ctx):
        super().__init__()
        self.filepath = filepath
        self.folder   = folder
        self.ctx_app  = ctx['app']
        self.ctx_url  = ctx['url']
        self.ctx_hour = ctx['hour']
        self._dismiss_timer = None
 
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFixedWidth(400)
 
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
 
        container = QWidget()
        container.setObjectName("container")
        container.setStyleSheet(self.STYLE)
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        layout.setContentsMargins(14, 12, 14, 12)
 
        # Header
        hdr = QHBoxLayout()
        ico = QLabel("📂"); ico.setStyleSheet("font-size:18px;")
        title_lbl = QLabel("AI File Organiser"); title_lbl.setObjectName("title")
        hdr.addWidget(ico); hdr.addWidget(title_lbl); hdr.addStretch()
        layout.addLayout(hdr)
 
        fn_lbl = QLabel(os.path.basename(filepath))
        fn_lbl.setObjectName("filename"); fn_lbl.setWordWrap(True)
        layout.addWidget(fn_lbl)
 
        bar = QProgressBar()
        bar.setMaximum(100); bar.setValue(int(confidence * 100))
        layout.addWidget(bar)
 
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244;")
        layout.addWidget(sep)
 
        if top_matches:
            layout.addWidget(QLabel("Move to:" if confidence >= 0.40 else "Where should this go?"))
            for i, (match_folder, match_conf) in enumerate(top_matches):
                short = Path(match_folder).name
                pct   = f"{match_conf*100:.0f}%"
                label = f"  📁  {short}   ({pct})"
                btn   = QPushButton(label)
                btn.setObjectName("btn_top" if i == 0 else "btn_alt")
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(lambda checked, f=match_folder: self._pick(f))
                layout.addWidget(btn)
        else:
            layout.addWidget(QLabel("Learning Mode — choose a folder to teach the AI:"))
 
        btn_browse = QPushButton("  🗂  Browse for another folder…")
        btn_browse.setObjectName("btn_browse")
        btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse.clicked.connect(self._browse_clicked)
        layout.addWidget(btn_browse)
 
        outer.addWidget(container)
 
        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self.on_dismiss)
        self._dismiss_timer.start(20000)
 
    def _pick(self, folder: str):
        if self._dismiss_timer:
            self._dismiss_timer.stop()
        if folder == self.folder:
            self.accepted.emit(self.filepath, folder, self.ctx_app, self.ctx_url, self.ctx_hour)
        else:
            self.rejected.emit(self.filepath, folder, self.ctx_app, self.ctx_url, self.ctx_hour)
        self.close()
 
    def _browse_clicked(self):
        """Stop dismiss timer, then open the folder dialog after a brief delay
        so the FramelessWindowHint window releases focus first (Windows fix)."""
        if self._dismiss_timer:
            self._dismiss_timer.stop()
        QTimer.singleShot(150, self._open_folder_dialog)
 
    def _open_folder_dialog(self):
        # Parent to None so Windows doesn't steal/hide it behind the frameless widget
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select Destination Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self.rejected.emit(self.filepath, folder, self.ctx_app, self.ctx_url, self.ctx_hour)
            self.close()
        else:
            # User cancelled — restart dismiss timer
            if self._dismiss_timer:
                self._dismiss_timer.start(20000)
 
    def on_dismiss(self):
        if self.isVisible():
            self.dismissed.emit(self.filepath)
            self.close()
 
    def show_animated(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.adjustSize()
        x     = screen.width()  - self.width()  - 25
        y_end = screen.height() - self.height() - 25
        self.move(x, screen.height() + 100)
        self.show()
        self.anim = QPropertyAnimation(self, b"pos")
        self.anim.setDuration(350)
        self.anim.setStartValue(QPoint(x, screen.height() + 100))
        self.anim.setEndValue(QPoint(x, y_end))
        self.anim.start()