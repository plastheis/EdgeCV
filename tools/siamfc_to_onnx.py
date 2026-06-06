"""Convert a huanglianghua/siamfc-pytorch AlexNetV1 checkpoint to ONNX.

Host-only tooling (ARCHITECTURE.md §11); NOT a runtime dependency. Requires the
[dev] extra (torch, onnx) and onnxruntime (in [test]) for the parity check.
Vendors a self-contained AlexNetV1 backbone + batch-1 cross-correlation head so
the original repo need not be installed.

Usage:
    python tools/siamfc_to_onnx.py \
        --checkpoint models/siamfc_alexnet_e50.pth \
        --out models/siamfc_generic.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bn(c: int) -> nn.BatchNorm2d:
    # huanglianghua uses eps=1e-6, momentum=0.05; eval-time eps affects numerics.
    return nn.BatchNorm2d(c, eps=1e-6, momentum=0.05)


class AlexNetV1(nn.Module):
    """Backbone matching siamfc-pytorch state_dict keys (backbone.conv1..conv5).
    Conv layers keep their default bias (the checkpoint stores conv biases)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2), _bn(96), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2), _bn(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1), _bn(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2), _bn(384), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class SiamFCHead(nn.Module):
    """Batch-1 cross-correlation: conv2d(search_feat, exemplar_feat) * out_scale.
    Numerically identical to the repo's _fast_xcorr for batch size 1 (the export
    and the edgecv tracker both run one exemplar against one search per call)."""

    def __init__(self, out_scale: float = 0.001) -> None:
        super().__init__()
        self.out_scale = out_scale

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, z) * self.out_scale


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = AlexNetV1()
        self.head = SiamFCHead()

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(z), self.backbone(x))


def build_net(state_dict: dict | None = None) -> Net:
    """Build the net; load a checkpoint state_dict strict=True when provided."""
    net = Net()
    if state_dict is not None:
        net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def export_onnx(net: Net, out_path: str, opset: int = 13) -> None:
    z = torch.zeros(1, 3, 127, 127)
    x = torch.zeros(1, 3, 255, 255)
    torch.onnx.export(
        net, (z, x), out_path,
        input_names=["exemplar", "search"], output_names=["score_map"],
        opset_version=opset, do_constant_folding=True)


def _parity_check(net: Net, out_path: str, tol: float = 1e-3) -> float:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    z = rng.standard_normal((1, 3, 127, 127)).astype(np.float32)
    x = rng.standard_normal((1, 3, 255, 255)).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.from_numpy(z), torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(["score_map"], {"exemplar": z, "search": x})[0]
    diff = float(np.max(np.abs(ref - got)))
    if diff > tol:
        raise SystemExit(f"parity check FAILED: max|delta|={diff:.2e} > {tol:.0e}")
    return diff


def convert(checkpoint: str, out: str) -> str:
    sd = torch.load(checkpoint, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:   # tolerate wrapped checkpoints
        sd = sd["state_dict"]
    net = build_net(sd)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    export_onnx(net, out)
    diff = _parity_check(net, out)
    print(f"exported {out}  (parity max|delta|={diff:.2e})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SiamFC PyTorch -> ONNX")
    ap.add_argument("--checkpoint", required=True, help="path to the .pth state_dict")
    ap.add_argument("--out", default="models/siamfc_generic.onnx")
    args = ap.parse_args()
    convert(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
