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


def test_nanotrack_manifest_loads():
    m = load_manifest(MANIFESTS / "nanotrack.yaml")
    assert m.name == "nanotrack"
    assert m.task == "sot_template_matching"
    assert {i["name"] for i in m.inputs} == {"exemplar", "search"}
    assert [o["name"] for o in m.outputs] == ["cls", "loc"]
    assert m.preprocessing["penalty_k"] == 0.138
    assert m.preprocessing["window_influence"] == 0.455
    # Split two-model artifacts: each half carries per-backend paths + its own io.
    bb, hd = m.artifacts["backbone"], m.artifacts["head"]
    assert bb["onnx"]["path"] == "nanotrack_backbone.onnx"
    assert bb["rknn"]["path"] == "nanotrack_backbone.rknn"
    assert hd["onnx"]["path"] == "nanotrack_head.onnx"
    assert hd["rknn"]["path"] == "nanotrack_head.rknn"
    assert [o["name"] for o in hd["io"]["outputs"]] == ["output1", "output2"]
