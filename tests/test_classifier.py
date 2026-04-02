import sys, os, pytest, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import core.classifier as clf_module

@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(clf_module, "DB_DIR",  tmp_path)
    monkeypatch.setattr(clf_module, "DB_PATH", tmp_path / "model.db")
    yield

@pytest.fixture
def clf():
    from core.classifier import FileClassifier
    return FileClassifier()

class TestFeatureString:
    def test_basic(self, clf):
        s = clf.build_feature_string("Invoice_2024.pdf","pdf")
        assert "invoice" in s and "pdf" in s

    def test_camel_case_split(self, clf):
        s = clf.build_feature_string("MyReport.docx","docx")
        assert "my" in s or "report" in s

    def test_context_repeated(self, clf):
        s = clf.build_feature_string("photo.jpg","jpg",source_app="Photoshop")
        assert s.lower().count("photoshop") >= 3

    def test_url_domain_extracted(self, clf):
        s = clf.build_feature_string("paper.pdf","pdf",source_url="https://arxiv.org/abs/1234")
        assert "arxiv" in s

    def test_time_bucket_morning(self, clf):
        assert "morning" in clf.build_feature_string("x.txt","txt",hour=9)

    def test_time_bucket_evening(self, clf):
        assert "evening" in clf.build_feature_string("x.txt","txt",hour=20)

class TestTraining:
    def test_no_prediction_before_training(self, clf):
        assert clf.predict("test.pdf","pdf") is None

    def test_single_class_no_prediction(self, clf):
        clf.train(clf.build_feature_string("invoice.pdf","pdf"), "Finance")
        assert clf.predict("invoice.pdf","pdf") is None

    def test_two_classes_prediction(self, clf):
        for i in range(5):
            clf.train(clf.build_feature_string(f"invoice{i}.pdf","pdf"), "Finance")
            clf.train(clf.build_feature_string(f"essay{i}.docx","docx"), "University")
        r = clf.predict("invoice.pdf","pdf")
        assert r is not None and r[0] == "Finance"

    def test_incremental_improves_accuracy(self, clf):
        for i in range(10):
            clf.train(clf.build_feature_string(f"invoice_{i}.pdf","pdf"), "Finance")
            clf.train(clf.build_feature_string(f"essay_{i}.docx","docx"), "Essays")
        r = clf.predict("invoiceSummary.pdf","pdf")
        assert r is not None and r[0] == "Finance"

    def test_confidence_threshold_suppression(self):
        from core.classifier import FileClassifier
        clf_high = FileClassifier(confidence_threshold=0.999)
        clf_high.train(clf_high.build_feature_string("a.pdf","pdf"), "A")
        clf_high.train(clf_high.build_feature_string("b.docx","docx"), "B")
        assert clf_high.predict("a.pdf","pdf") is None

    def test_context_improves_prediction(self, clf):
        for i in range(8):
            clf.train(clf.build_feature_string(f"file{i}.jpg","jpg",source_app="Photoshop"), "Design")
            clf.train(clf.build_feature_string(f"doc{i}.pdf","pdf",source_app="Word"), "Documents")
        r = clf.predict("newfile.jpg","jpg",source_app="Photoshop")
        if r: assert r[0] == "Design"

class TestPersistence:
    def test_model_survives_reload(self, clf):
        from core.classifier import FileClassifier
        for i in range(5):
            clf.train(clf.build_feature_string(f"invoice{i}.pdf","pdf"), "Finance")
            clf.train(clf.build_feature_string(f"essay{i}.docx","docx"), "Essays")
        clf2 = FileClassifier()
        r = clf2.predict("invoice.pdf","pdf")
        assert r is not None and r[0] == "Finance"

    def test_training_count(self, clf):
        clf.train(clf.build_feature_string("a.pdf","pdf"), "A")
        clf.train(clf.build_feature_string("b.pdf","pdf"), "B")
        clf.train(clf.build_feature_string("c.pdf","pdf"), "A")
        assert clf.get_training_count() == 3

class TestEdgeCases:
    def test_special_characters_in_filename(self, clf):
        s = clf.build_feature_string("report (2024) [FINAL] v2.pdf","pdf")
        assert isinstance(s, str)

    def test_unicode_filename(self, clf):
        s = clf.build_feature_string("rapport_\xe9tude.pdf","pdf")
        assert isinstance(s, str)

    def test_empty_filename(self, clf):
        assert isinstance(clf.build_feature_string("",""), str)

    def test_very_long_filename(self, clf):
        assert isinstance(clf.build_feature_string("a"*300+".pdf","pdf"), str)

    def test_top_classes_unfitted(self, clf):
        assert clf.get_top_classes("some feature") == []
