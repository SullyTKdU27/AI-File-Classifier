import sys, logging, argparse
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon
from app import AppController
 
def main():
    p = argparse.ArgumentParser(description="AI File Classifier")
    p.add_argument("--watch",   nargs="*", metavar="DIR",
                   help="Directories to watch for new files (default: Downloads + Desktop)")
    p.add_argument("--folders", nargs="*", metavar="FOLDER",
                   help="Known destination folders to seed the AI with so it starts predicting immediately")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    args = p.parse_args()
 
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%H:%M:%S"
    )
 
    watch_dirs   = [Path(d).expanduser().resolve() for d in args.watch]   if args.watch   else None
    seed_folders = [Path(d).expanduser().resolve() for d in args.folders] if args.folders else None
 
    app = QApplication(sys.argv)
    app.setApplicationName("AI File Classifier")
    app.setQuitOnLastWindowClosed(False)
 
    ctrl = AppController(watch_dirs=watch_dirs, seed_folders=seed_folders)
    ctrl.start(app)
 
    print("AI File Classifier running.")
    if seed_folders:
        print(f"  Seeded {len(seed_folders)} folders: {[f.name for f in seed_folders]}")
    print("  Right-click the tray icon for Stats / Settings.")
    sys.exit(app.exec())
 
if __name__ == "__main__":
    main()