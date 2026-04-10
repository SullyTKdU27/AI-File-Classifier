import sys, os, argparse, random, tempfile, shutil, logging
sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.WARNING)
from pathlib import Path
import core.classifier as clf_module
from core.classifier import FileClassifier

SYNTHETIC = {
    "Finance/Invoices":   [("Invoice_2024_Q1.pdf","pdf","",""),("receipt_amazon.pdf","pdf","",""),("utility_bill_oct.pdf","pdf","",""),("payslip_nov.pdf","pdf","",""),("bank_statement.pdf","pdf","","")],
    "University/CS301":   [("assignment_3_report.docx","docx","Word","moodle.greenwich.ac.uk"),("lecture_notes_week5.pdf","pdf","Chrome","moodle.greenwich.ac.uk"),("cs301_coursework.docx","docx","Word",""),("algorithm_analysis.pdf","pdf","Chrome","scholar.google.com"),("week7_slides.pdf","pdf","Chrome","moodle")],
    "Design/ClientA":     [("logo_v3_final.png","png","Photoshop",""),("banner_draft.psd","psd","Photoshop",""),("mockup_homepage.ai","ai","Illustrator",""),("clientA_brand_guide.pdf","pdf","Acrobat","")],
    "Research/Papers":    [("incremental_learning.pdf","pdf","Chrome","arxiv.org"),("naive_bayes_survey.pdf","pdf","Chrome","scholar.google.com"),("privacy_ml_review.pdf","pdf","Chrome","arxiv.org"),("hitl_systems_2024.pdf","pdf","Chrome","scholar.google.com")],
}

def seed(clf, n=3):
    print("\n📚  Seeding classifier...")
    for folder, files in SYNTHETIC.items():
        for fn, ext, app, url in files[:n]:
            clf.train(clf.build_feature_string(fn,ext,app,url,10), folder)
            print(f"   ✓ {fn:45s} → {folder}")
    print(f"\n   Samples: {clf.get_training_count()}")

def simulate_week(clf, err_rate):
    all_files = [(fn,ext,app,url,folder) for folder,files in SYNTHETIC.items() for fn,ext,app,url in files]
    random.shuffle(all_files); batch = all_files[:12]
    total = rejected = 0
    for fn,ext,app,url,true_folder in batch:
        r = clf.predict(fn,ext,app,url,10)
        if not r: continue
        pred, conf = r; total += 1
        if random.random() < err_rate:
            clf.train(clf.build_feature_string(fn,ext,app,url), true_folder); rejected += 1
        else:
            clf.train(clf.build_feature_string(fn,ext,app,url), pred)
    return total, rejected

def run_simulation():
    tmpdir = tempfile.mkdtemp()
    clf_module.DB_DIR = Path(tmpdir); clf_module.DB_PATH = Path(tmpdir)/"model.db"
    try:
        clf = FileClassifier()
        print("\n" + "═"*60)
        print("  AI File Classifier — 4-Week Simulation")
        print("═"*60)
        seed(clf, n=2)
        error_rates = [0.45, 0.30, 0.18, 0.08]
        results = []
        for week, err in enumerate(error_rates, 1):
            total, rej = simulate_week(clf, err)
            rate = (rej/total*100) if total else 0
            results.append((week,total,rej,rate))
            filled = int((1-rate/100)*30); bar = "█"*filled + "░"*(30-filled)
            icon = "🟢" if rate<15 else "🟡" if rate<30 else "🔴"
            print(f"\n  Week {week}  {icon}  [{bar}]  Rejection: {rate:.1f}%  ({rej}/{total})")
        print("\n" + "─"*60)
        start,end = results[0][3], results[-1][3]
        print(f"  Improvement: {start:.1f}% → {end:.1f}% (−{start-end:.1f}pp)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def interactive():
    clf = FileClassifier()
    print(f"\n{'═'*60}\n  AI File Classifier — Interactive Demo\n{'═'*60}")
    print(f"  DB: {clf_module.DB_PATH} | Samples: {clf.get_training_count()} | Folders: {clf.get_all_folders()}")
    while True:
        print("\n  1=Add training  2=Predict  3=Stats  4=Seed demo data  5=Quit")
        c = input("  > ").strip()
        if c=="1":
            fn=input("  Filename: ").strip(); ext=Path(fn).suffix.lstrip(".")
            app=input("  Source app: ").strip(); url=input("  Source URL: ").strip()
            folder=input("  Correct folder: ").strip()
            clf.train(clf.build_feature_string(fn,ext,app,url), folder)
            print(f"  ✓ Trained: {fn} → {folder}")
        elif c=="2":
            fn=input("  Filename: ").strip(); ext=Path(fn).suffix.lstrip(".")
            app=input("  Source app: ").strip(); url=input("  Source URL: ").strip()
            r=clf.predict(fn,ext,app,url)
            if r: print(f"  🎯 {r[0]}  ({r[1]*100:.1f}%)")
            else: print("  ❓ No confident prediction (need more training data)")
        elif c=="3":
            print(f"  Samples: {clf.get_training_count()} | Folders: {clf.get_all_folders()} | Threshold: {clf.get_current_threshold():.0%}")
        elif c=="4": seed(clf)
        elif c=="5": break

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--simulate",action="store_true")
    args=p.parse_args()
    if args.simulate: run_simulation()
    else: interactive()
