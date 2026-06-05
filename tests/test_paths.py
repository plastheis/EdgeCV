import os

from edgecv.models.paths import resolve_artifact_path


def test_absolute_path_passes_through(tmp_path):
    p = tmp_path / "model.onnx"
    assert resolve_artifact_path(str(p)) == str(p)


def test_relative_resolves_against_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EDGECV_MODEL_DIR", str(tmp_path))
    assert resolve_artifact_path("siamfc_generic.onnx") == str(tmp_path / "siamfc_generic.onnx")


def test_relative_default_is_models_dir(monkeypatch):
    monkeypatch.delenv("EDGECV_MODEL_DIR", raising=False)
    expected = os.path.join("models", "siamfc_generic.onnx")
    assert resolve_artifact_path("siamfc_generic.onnx") == expected
