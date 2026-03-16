import sys, logging, argparse
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon
from app import AppController

def main():
    p = argparse.ArgumentParser(description="AI File Classifier")
    p.add_argument("--watch", nargs="*", metavar="DIR")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)-8s %(name)s %(message)s", datefmt="%H:%M:%S")
    watch_dirs = [Path(d).expanduser().resolve() for d in args.watch] if args.watch else None
    app = QApplication(sys.argv)
    app.setApplicationName("AI File Classifier")
    app.setQuitOnLastWindowClosed(False)
    ctrl = AppController(watch_dirs=watch_dirs)
    ctrl.start(app)
    print("AI File Classifier running. Right-click tray icon to access settings.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
