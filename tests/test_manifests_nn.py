from pathlib import Path

from edgecv.models.manifest import load_manifest

MANIFESTS = Path("edgecv/models/manifests")


def test_siamfc_manifest_loads():
    m = load_manifest(MANIFESTS / "siamfc_generic.yaml")
    assert m.name == "siamfc_generic"
    assert {i["name"] for i in m.inputs} == {"exemplar", "search"}
    assert m.outputs[0]["name"] == "score_map"
    assert m.preprocessing["color"] == "rgb"


def test_yolo26n_manifest_loads():
    m = load_manifest(MANIFESTS / "yolo26n.yaml")
    assert m.name == "yolo26n"
    assert m.task == "detection"
    assert m.preprocessing["class_agnostic"] is True
    assert m.preprocessing["output_format"] == "yolov8"
    assert m.outputs[0]["shape"] == [1, 84, -1]


def test_yolo26s_manifest_loads():
    m = load_manifest(MANIFESTS / "yolo26s.yaml")
    assert m.name == "yolo26s"
    assert m.preprocessing["output_format"] == "yolov8"
    assert m.artifacts["onnx"]["path"] == "yolo26s.onnx"


def test_yolo_generic_is_retired():
    assert not (MANIFESTS / "yolo_generic.yaml").exists()
