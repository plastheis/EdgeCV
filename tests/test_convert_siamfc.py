import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

_PATH = Path(__file__).resolve().parent.parent / "tools" / "siamfc_to_onnx.py"
_spec = importlib.util.spec_from_file_location("siamfc_to_onnx", _PATH)
conv = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(conv)


def test_build_net_loads_strict_and_runs():
    net = conv.build_net()
    z = torch.zeros(1, 3, 127, 127)
    x = torch.zeros(1, 3, 255, 255)
    with torch.no_grad():
        out = net(z, x)
    assert tuple(out.shape) == (1, 1, 17, 17)


def test_roundtrip_checkpoint_to_onnx_parity(tmp_path):
    net = conv.build_net()                       # random init
    ckpt = tmp_path / "fake.pth"
    torch.save(net.state_dict(), ckpt)
    out = tmp_path / "siamfc.onnx"
    conv.convert(str(ckpt), str(out))            # loads strict=True, exports, parity-checks
    assert out.exists()

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    z = rng.standard_normal((1, 3, 127, 127)).astype(np.float32)
    x = rng.standard_normal((1, 3, 255, 255)).astype(np.float32)
    got = sess.run(["score_map"], {"exemplar": z, "search": x})[0]
    with torch.no_grad():
        ref = conv.build_net(torch.load(ckpt))(
            torch.from_numpy(z), torch.from_numpy(x)).numpy()
    assert got.shape == (1, 1, 17, 17)
    assert float(np.max(np.abs(ref - got))) < 1e-3
