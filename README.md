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

Conversion runs **offline on x86**; the device only ever runs the lite runtime. One
manifest-driven dispatcher (`tools/convert.py`) converts any registered model:

```bash
pip install -e .[dev]          # torch + onnx for export/validation

# PyTorch checkpoint -> ONNX (writes the manifest's resolved artifact path under models/)
python tools/convert.py --model siamfc_generic --checkpoint models/siamfc_alexnet_e50.pth

# ...and chain to an RK3588 INT8 RKNN (needs rknn-toolkit2 + calibration images)
python tools/convert.py --model siamfc_generic --checkpoint models/siamfc_alexnet_e50.pth \
    --rknn --calib calib/
```

Conversion is three stages; only the first is model-specific: load checkpoint →
`nn.Module` (a per-model **adapter**), export → ONNX (generic harness: export +
`onnx.checker` + torch-vs-onnxruntime parity), ONNX → RKNN (generic, optional). I/O names
and shapes come from the model's manifest, so the converter and the runtime backend never
disagree.

**Adding a new model** = one ~20-line adapter (`tools/convert_lib/adapters/<name>.py`) that
turns a checkpoint into a loaded module, registered against a manifest — then
`python tools/convert.py --model <name> --checkpoint <pth>`. Full mechanics, the
add-a-tracker recipe, the preprocessing contract, and INT8 calibration notes are in
**[`tools/CONVERSION.md`](tools/CONVERSION.md)**.

## Status

Foundation only — no concrete trackers yet. See
`docs/superpowers/specs/2026-05-31-edgecv-foundation-design.md`.
