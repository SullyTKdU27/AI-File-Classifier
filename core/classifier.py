import re, sqlite3, pickle, logging, threading, warnings
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
DB_DIR  = Path.home() / ".ai_file_classifier"
DB_PATH = DB_DIR / "model.db"
BASE_THRESHOLD = 0.40

def _adaptive_threshold(n: int) -> float:
    if n < 2: return BASE_THRESHOLD
    return max(BASE_THRESHOLD, (1.0 / n) + 0.15)

class FileClassifier:
    def __init__(self, confidence_threshold: float = None):
        self._lock = threading.Lock()
        self._fixed_threshold = confidence_threshold
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self._db = self._init_db()
        self._load_model()

    def _init_db(self):
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS training_log (id INTEGER PRIMARY KEY AUTOINCREMENT, feature_str TEXT NOT NULL, folder TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS model_blob (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS folder_registry (folder TEXT PRIMARY KEY, count INTEGER DEFAULT 0)")
        conn.commit()
        return conn

    def _make_vectorizer(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TfidfVectorizer(analyzer="word", tokenizer=self._tokenize, token_pattern=None, ngram_range=(1,2), min_df=1, sublinear_tf=True)

    def _load_model(self):
        row = self._db.execute("SELECT value FROM model_blob WHERE key='model'").fetchone()
        if row:
            try:
                s = pickle.loads(row[0])
                self._vectorizer = s["vectorizer"]; self._clf = s["clf"]
                self._classes = s["classes"]; self._is_fitted = s.get("is_fitted", False)
                return
            except Exception as e:
                logger.warning("Could not load model: %s", e)
        self._vectorizer = self._make_vectorizer()
        self._clf = MultinomialNB(alpha=0.5)
        self._classes = []; self._is_fitted = False

    def _save_model(self):
        blob = pickle.dumps({"vectorizer": self._vectorizer, "clf": self._clf, "classes": self._classes, "is_fitted": self._is_fitted})
        self._db.execute("INSERT OR REPLACE INTO model_blob (key,value) VALUES ('model',?)", (blob,))
        self._db.commit()

    @staticmethod
    def _tokenize(text: str) -> list:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        stops = {"the","a","an","of","in","for","to","and","or"}
        tokens = [t for t in tokens if t not in stops and len(t) > 1]
        if tokens: tokens = [tokens[0]] * 3 + tokens
        return tokens

    @staticmethod
    def build_feature_string(filename:str, extension:str, source_app:str="", source_url:str="", hour:int=12) -> str:
        stem = Path(filename).stem
        if   6  <= hour < 12: tb = "morning"
        elif 12 <= hour < 18: tb = "afternoon"
        elif 18 <= hour < 23: tb = "evening"
        else:                  tb = "night"
        url_domain = ""
        if source_url:
            m = re.search(r"(?:https?://)?(?:www\.)?([^/\s?#]+)", source_url)
            if m: url_domain = m.group(1).replace(".", " ")
        parts = [stem, extension.lstrip("."), tb]
        if source_app: parts += [source_app] * 3
        if url_domain: parts += [url_domain] * 3
        return " ".join(parts)

    def train(self, feature_str: str, folder: str):
        with self._lock:
            self._db.execute("INSERT INTO training_log (feature_str,folder) VALUES (?,?)", (feature_str, folder))
            self._db.execute("INSERT INTO folder_registry (folder,count) VALUES (?,1) ON CONFLICT(folder) DO UPDATE SET count=count+1", (folder,))
            self._db.commit()
            if folder not in self._classes: self._classes.append(folder)
            rows = self._db.execute("SELECT feature_str,folder FROM training_log").fetchall()
            X_str = [r[0] for r in rows]; y = [r[1] for r in rows]
            if len(set(y)) < 2: self._save_model(); return
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._vectorizer = self._make_vectorizer()
                X = self._vectorizer.fit_transform(X_str)
            self._clf = MultinomialNB(alpha=0.5); self._clf.fit(X, y)
            self._is_fitted = True; self._save_model()

    def predict(self, filename:str, extension:str="", source_app:str="", source_url:str="", hour:int=12) -> Optional[tuple]:
        with self._lock:
            if not self._is_fitted or not self._classes: return None
            feat = self.build_feature_string(filename, extension, source_app, source_url, hour)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X = self._vectorizer.transform([feat])
                probs = self._clf.predict_proba(X)[0]
                best = int(np.argmax(probs))
                conf = float(probs[best]); folder = self._clf.classes_[best]
                thresh = self._fixed_threshold if self._fixed_threshold is not None else _adaptive_threshold(len(self._classes))
                if conf < thresh: return None
                return folder, conf
            except Exception as e:
                logger.error("Prediction error: %s", e); return None

    def get_top_classes(self, feature_str:str, n:int=3) -> list:
        if not self._is_fitted: return []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = self._vectorizer.transform([feature_str])
            probs = self._clf.predict_proba(X)[0]
            pairs = sorted(zip(self._clf.classes_, probs), key=lambda x: x[1], reverse=True)
            return [(c, float(p)) for c,p in pairs[:n]]
        except: return []

    def get_all_folders(self) -> list:
        return [r[0] for r in self._db.execute("SELECT folder FROM folder_registry ORDER BY count DESC").fetchall()]

    def get_training_count(self) -> int:
        r = self._db.execute("SELECT COUNT(*) FROM training_log").fetchone()
        return r[0] if r else 0

    def get_current_threshold(self) -> float:
        return self._fixed_threshold if self._fixed_threshold is not None else _adaptive_threshold(len(self._classes))

    def get_rejection_stats(self) -> dict:
        rows = self._db.execute("SELECT strftime('%W-%Y',timestamp) as week, COUNT(*) as total FROM training_log GROUP BY week ORDER BY week").fetchall()
        return {r[0]: r[1] for r in rows}
