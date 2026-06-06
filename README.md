# edgecv

Single-object visual trackers scoped to real-time deployment on edge hardware.
See `ARCHITECTURE.md` for the design.

## Install

```bash
pip install edgecv             # core: numpy CF runtime, fusion abstractions, mock backend
pip install edgecv[onnx]       # ONNXRuntime CPU/dev backend
pip install edgecv[rknn]       # registers the RKNN backend (see device note below)
pip install edgecv[test]       # test + lint tooling
```

### RKNN on-device note

`rknn-toolkit-lite2` is **not on PyPI** and is **installed manually on the device**
(Rockchip release archive). The `[rknn]` extra only registers the backend adapter;
it does not and cannot pull the runtime.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .[onnx,test]
pytest -q
```

## Models & conversion

Trackers never load a weight file directly — they load a **manifest**
(`edgecv/models/manifests/*.yaml`) that maps one logical model to per-backend
artifacts plus its preprocessing/IO spec (ARCHITECTURE.md §10.1). Swapping a
PyTorch export for an INT8 `.rknn` is a manifest change, not a code change.

### Where models go

- Weight blobs live in **`models/`** at the repo root. They are **gitignored**
  and never committed (host-only, often large or board-specific).
- A manifest's `artifacts.<backend>.path` is **relative** and resolves against
  **`$EDGECV_MODEL_DIR`** (default `models/`); absolute paths pass through. So
  from the repo root, `artifacts: { onnx: { path: siamfc_generic.onnx } }` finds
  `models/siamfc_generic.onnx`. Point `EDGECV_MODEL_DIR` elsewhere to relocate.

### The conversion pipeline (host-only)

Conversion runs **offline on x86**; the device only ever runs the lite runtime.
Install the tooling deps and use the scripts in `tools/`:

```bash
pip install -e .[dev]          # torch + onnx for export/validation
```

| Step | Tool | Output |
|------|------|--------|
| PyTorch → **ONNX** (portable; x86 dev + CI) | per-model script, e.g. `tools/siamfc_to_onnx.py` | `models/<name>.onnx` |
| ONNX → **RKNN** (on-device NPU, INT8) | `tools/onnx_to_rknn.py` (needs `rknn-toolkit2` + calibration images) | `models/<name>.rk3588.rknn` |

```bash
# example: SiamFC checkpoint -> ONNX (parity-checked against the torch model)
python tools/siamfc_to_onnx.py --checkpoint models/siamfc_alexnet_e50.pth \
    --out models/siamfc_generic.onnx

# example: ONNX -> RKNN for the RK3588 NPU (run on an x86 host with rknn-toolkit2)
python tools/onnx_to_rknn.py --onnx models/siamfc_generic.onnx \
    --out models/siamfc_generic.rk3588.rknn --target rk3588 \
    --calibration-dir calib/ --inputs exemplar search
```

**Adding a new model** follows the same shape: write a host script under `tools/`
that emits a backend artifact into `models/`, then add (or extend) a manifest that
points at it. Full command reference, the preprocessing contract, and INT8
calibration notes are in **[`tools/CONVERSION.md`](tools/CONVERSION.md)**.

## Status

Foundation only — no concrete trackers yet. See
`docs/superpowers/specs/2026-05-31-edgecv-foundation-design.md`.
