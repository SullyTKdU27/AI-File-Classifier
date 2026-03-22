from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QProgressBar
from PyQt6.QtCore    import Qt, pyqtSignal, QPropertyAnimation, QPoint, QTimer
import os

class NotificationWidget(QWidget):
    accepted  = pyqtSignal(str, str, str, str, int)
    rejected  = pyqtSignal(str, str, str, str, int)
    dismissed = pyqtSignal(str)

    def __init__(self, filepath, folder, confidence, top_classes, ctx):
        super().__init__()
        self.filepath = filepath
        self.folder = folder
        # Store context for training signal
        self.ctx_app = ctx['app']
        self.ctx_url = ctx['url']
        self.ctx_hour = ctx['hour']
        
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        self.setFixedWidth(380)
        layout = QVBoxLayout(self)
        container = QWidget(); container.setObjectName("container")
        container.setStyleSheet("""
            QWidget#container { background-color: #1e1e1e; border: 2px solid #333; border-radius: 15px; }
            QLabel { color: white; font-family: 'Segoe UI'; }
            QPushButton { background-color: #0078d4; color: white; border-radius: 6px; padding: 10px; font-weight: bold; }
            QPushButton#btn_choose { background-color: #3d3d3d; }
        """)
        container_layout = QVBoxLayout(container)

        container_layout.addWidget(QLabel(f"📂 New File: {os.path.basename(filepath)}"))

        if confidence == 0:
            msg, bar_color = "<b>Learning Mode</b><br>Where should this go?", "#444"
        elif confidence < 0.40:
            msg, bar_color = f"AI Guessing: <b>{os.path.basename(folder)}</b>", "#d43f3a"
        else:
            msg, bar_color = f"Smart Match: <b>{os.path.basename(folder)}</b>", "#0078d4"

        suggestion = QLabel(msg); suggestion.setWordWrap(True)
        container_layout.addWidget(suggestion)

        self.bar = QProgressBar(); self.bar.setMaximum(100); self.bar.setValue(int(confidence * 100))
        self.bar.setStyleSheet(f"QProgressBar {{ height: 6px; background: #333; border: none; }} QProgressBar::chunk {{ background: {bar_color}; }}")
        container_layout.addWidget(self.bar)

        btn_layout = QHBoxLayout()
        self.btn_move = QPushButton("✓ Move Here")
        self.btn_choose = QPushButton("✗ Choose Folder")
        self.btn_choose.setObjectName("btn_choose")
        btn_layout.addWidget(self.btn_move); btn_layout.addWidget(self.btn_choose)
        container_layout.addLayout(btn_layout)
        layout.addWidget(container)

        self.btn_move.clicked.connect(self.on_accept)
        self.btn_choose.clicked.connect(self.on_choose)
        QTimer.singleShot(15000, self.on_dismiss)

    def on_accept(self):
        if self.bar.value() == 0: self.on_choose()
        else:
            self.accepted.emit(self.filepath, self.folder, self.ctx_app, self.ctx_url, self.ctx_hour)
            self.close()

    def on_choose(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.rejected.emit(self.filepath, folder, self.ctx_app, self.ctx_url, self.ctx_hour)
            self.close()

    def on_dismiss(self):
        self.dismissed.emit(self.filepath); self.close()

    def show_animated(self):
        screen = QApplication.primaryScreen().availableGeometry()
        x, y_end = screen.width() - self.width() - 25, screen.height() - self.height() - 25
        self.move(x, screen.height() + 100); self.show()
        self.anim = QPropertyAnimation(self, b"pos"); self.anim.setDuration(400)
        self.anim.setStartValue(QPoint(x, screen.height() + 100)); self.anim.setEndValue(QPoint(x, y_end))
        self.anim.start()