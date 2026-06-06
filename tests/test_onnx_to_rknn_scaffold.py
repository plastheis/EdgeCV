import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "tools" / "onnx_to_rknn.py"
_spec = importlib.util.spec_from_file_location("onnx_to_rknn", _PATH)
rk = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rk)


def test_module_exposes_main_and_convert():
    # Scaffold imports without rknn-toolkit2 present (guarded import).
    assert hasattr(rk, "main")
    assert hasattr(rk, "convert")
