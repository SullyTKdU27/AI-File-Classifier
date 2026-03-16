from pathlib import Path
from PyQt6.QtCore    import Qt
from PyQt6.QtGui     import QIcon, QPixmap, QPainter, QColor, QBrush
from PyQt6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu,
                              QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QListWidget, QListWidgetItem,
                              QSlider, QGroupBox, QFileDialog)

STYLE = "QDialog{background:#1e1e2e;color:#cdd6f4;}QLabel{color:#cdd6f4;}QGroupBox{color:#89b4fa;border:1px solid #313244;border-radius:6px;margin-top:8px;padding:6px;}QGroupBox::title{subcontrol-origin:margin;left:8px;}QListWidget{background:#181825;border:1px solid #313244;border-radius:4px;color:#cdd6f4;}QPushButton{background:#313244;color:#cdd6f4;border-radius:6px;padding:5px 12px;}QPushButton:hover{background:#45475a;}"

def _make_icon(color="#89b4fa"):
    px = QPixmap(22,22); px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QBrush(QColor(color))); p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2,2,18,18); p.end(); return QIcon(px)

class TrayApp:
    def __init__(self, ctrl):
        self._ctrl = ctrl
        self._tray = QSystemTrayIcon(_make_icon())
        self._tray.setToolTip("AI File Classifier – running")
        self._tray.activated.connect(lambda r: self._show_stats() if r==QSystemTrayIcon.ActivationReason.DoubleClick else None)
        menu = QMenu()
        menu.addAction("📊  View Stats",  self._show_stats)
        menu.addAction("⚙️   Settings",    self._show_settings)
        menu.addSeparator()
        menu.addAction("❌  Quit",         self._quit)
        self._tray.setContextMenu(menu)

    def show(self):   self._tray.show()
    def notify(self, title, msg): self._tray.showMessage(title, msg, QSystemTrayIcon.MessageIcon.Information, 3000)
    def _show_stats(self):    StatsDialog(self._ctrl).exec()
    def _show_settings(self): SettingsDialog(self._ctrl).exec()
    def _quit(self):  self._ctrl.shutdown(); QApplication.quit()

class StatsDialog(QDialog):
    def __init__(self, ctrl):
        super().__init__(); self.setWindowTitle("Statistics")
        self.setMinimumSize(400,340); self.setStyleSheet(STYLE)
        lay = QVBoxLayout(self)
        clf = ctrl.classifier
        g1 = QGroupBox("Overview"); g1l = QVBoxLayout(g1)
        g1l.addWidget(QLabel(f"Training samples: <b>{clf.get_training_count()}</b>"))
        g1l.addWidget(QLabel(f"Known folders: <b>{len(clf.get_all_folders())}</b>"))
        g1l.addWidget(QLabel(f"Threshold: <b>{clf.get_current_threshold():.0%}</b>"))
        lay.addWidget(g1)
        g2 = QGroupBox("Known Folders"); g2l = QVBoxLayout(g2)
        lst = QListWidget()
        for f in clf.get_all_folders(): lst.addItem(QListWidgetItem(f"  📁  {f}"))
        g2l.addWidget(lst); lay.addWidget(g2)
        btn = QPushButton("Close"); btn.clicked.connect(self.accept); lay.addWidget(btn)

class SettingsDialog(QDialog):
    def __init__(self, ctrl):
        super().__init__(); self.setWindowTitle("Settings")
        self.setMinimumSize(440,300); self.setStyleSheet(STYLE); self._ctrl = ctrl
        lay = QVBoxLayout(self)
        g1 = QGroupBox("Watched Folders"); g1l = QVBoxLayout(g1)
        self._lst = QListWidget()
        for d in ctrl.watch_dirs: self._lst.addItem(str(d))
        g1l.addWidget(self._lst)
        br = QHBoxLayout()
        ab = QPushButton("＋ Add"); ab.clicked.connect(self._add)
        rb = QPushButton("－ Remove"); rb.clicked.connect(self._remove)
        br.addWidget(ab); br.addWidget(rb); g1l.addLayout(br); lay.addWidget(g1)
        g2 = QGroupBox("Confidence Threshold"); g2l = QVBoxLayout(g2)
        self._cl = QLabel(f"Current: {int(ctrl.threshold*100)}%")
        sl = QSlider(Qt.Orientation.Horizontal); sl.setRange(40,90); sl.setValue(int(ctrl.threshold*100))
        sl.valueChanged.connect(lambda v: (self._cl.setText(f"Current: {v}%"), setattr(ctrl,'threshold',v/100)))
        g2l.addWidget(self._cl); g2l.addWidget(sl); lay.addWidget(g2)
        btn = QPushButton("Save & Close"); btn.clicked.connect(self.accept); lay.addWidget(btn)

    def _add(self):
        f = QFileDialog.getExistingDirectory(self,"Select folder to watch")
        if f: self._lst.addItem(f); self._ctrl.add_watch_directory(f)
    def _remove(self):
        for item in self._lst.selectedItems(): self._lst.takeItem(self._lst.row(item))
