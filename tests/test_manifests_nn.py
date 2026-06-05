from pathlib import Path

from edgecv.models.manifest import load_manifest

MANIFESTS = Path("edgecv/models/manifests")


def test_siamfc_manifest_loads():
    m = load_manifest(MANIFESTS / "siamfc_generic.yaml")
    assert m.name == "siamfc_generic"
    assert {i["name"] for i in m.inputs} == {"exemplar", "search"}
    assert m.outputs[0]["name"] == "score_map"
    assert m.preprocessing["color"] == "rgb"


def test_yolo_manifest_loads():
    m = load_manifest(MANIFESTS / "yolo_generic.yaml")
    assert m.name == "yolo_generic"
    assert m.task == "detection"
    assert m.preprocessing["class_agnostic"] is True
    assert m.preprocessing["output_format"] == "yolov5"
