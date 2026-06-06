# Model conversion (host-only)

These tools are **host-only** and not runtime dependencies (ARCHITECTURE.md §11).
Conversion runs offline on x86; the device only ever runs the lite runtime.

## Install

```bash
pip install -e .[dev]      # torch, onnx  (+ onnxruntime from [test] for parity checks)
```

`rknn-toolkit2` (for the RKNN step) is **not on PyPI** — install it on an x86 host
from Rockchip's release wheels.

## SiamFC: PyTorch → ONNX

Weights: `huanglianghua/siamfc-pytorch` AlexNetV1 (e.g. `siamfc_alexnet_e50.pth`).
Place the checkpoint under `models/` (gitignored) and run:

```bash
python tools/siamfc_to_onnx.py \
    --checkpoint models/siamfc_alexnet_e50.pth \
    --out models/siamfc_generic.onnx
```

The tool vendors a self-contained AlexNetV1 backbone + batch-1 cross-correlation
head, loads the checkpoint with `strict=True` (a key mismatch fails loudly), exports
the single two-input graph `(exemplar[1,3,127,127], search[1,3,255,255]) →
score_map[1,1,17,17]`, and runs a torch-vs-onnxruntime parity check (`max|Δ| < 1e-3`).

**Preprocessing contract (matches training):** RGB 3-channel, **raw `[0,255]`** pixels
(no `/255`, no mean/std), **BGR** channel order (cv2 convention). This is encoded in
`edgecv/models/manifests/siamfc_generic.yaml` (`color: rgb`, `scale: 1.0`) and consumed
by `SiamFC` via manifest-preprocessing precedence. The caller feeds BGR frames; e.g.
`tools/track_webcam.py` reads frames with cv2 (already BGR).

## ONNX → RKNN (scaffold)

Run on an x86 host with `rknn-toolkit2` installed. INT8 quantisation needs a folder
of representative calibration images:

```bash
python tools/onnx_to_rknn.py \
    --onnx models/siamfc_generic.onnx \
    --out models/siamfc_generic.rk3588.rknn \
    --target rk3588 \
    --calibration-dir calib/ \
    --inputs exemplar search
```

INT8 quantisation noise in NPU-derived score maps is largely self-correcting once the
tracker runs (PSR gate / appearance robustness); start from `siamfc_generic.onnx` for
dev/CI on x86, add the RKNN artifact for on-device deployment.

## Running the artifact / paths

Backends resolve a relative `artifacts.<backend>.path` against `$EDGECV_MODEL_DIR`
(default `models/`); absolute paths pass through. So from the repo root:

```bash
python tools/track_webcam.py --tracker siamfc   # finds models/siamfc_generic.onnx
```

(Model blobs — `.pth`, `.onnx`, `.rknn` — are gitignored and never committed.)
