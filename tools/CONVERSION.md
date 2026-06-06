# Model conversion (host-only)

Host-only tooling (ARCHITECTURE.md §11), not a runtime dependency. Conversion runs
offline on x86; the device only ever runs the lite runtime. One dispatcher converts any
registered model, driven by that model's manifest.

## Install

```bash
pip install -e .[dev]      # torch, onnx (+ onnxruntime from [test] for parity checks)
```

`rknn-toolkit2` (for the RKNN step) is **not on PyPI** — install it on an x86 host from
Rockchip's release wheels.

## Where models go

- Weight blobs live in `models/` at the repo root, **gitignored**, never committed.
- A manifest's `artifacts.<backend>.path` is relative and resolves against
  `$EDGECV_MODEL_DIR` (default `models/`). The converter writes the artifact to that same
  resolved path, so the tracker loads exactly what you produced.

## Quick start

```bash
# PyTorch checkpoint -> ONNX (writes models/siamfc_generic.onnx)
python tools/convert.py --model siamfc_generic --checkpoint models/siamfc_alexnet_e50.pth

# ...and chain to an RK3588 INT8 RKNN (needs rknn-toolkit2 + calibration images)
python tools/convert.py --model siamfc_generic --checkpoint models/siamfc_alexnet_e50.pth \
    --rknn --calib calib/

# convert an ONNX produced elsewhere (e.g. ultralytics) straight to RKNN
python tools/onnx_to_rknn.py --onnx models/yolo.onnx --out models/yolo.rk3588.rknn \
    --target rk3588 --calibration-dir calib/ --inputs images
```

## How it works

Conversion is three stages; only the first is model-specific:

1. **Load checkpoint → `nn.Module`** — per-model *adapter*.
2. **Export module → ONNX** — generic *harness*: `torch.onnx.export` → `onnx.checker` →
   torch-vs-onnxruntime parity (`max|Δ| < 1e-3`).
3. **ONNX → RKNN** — generic, optional.

The **manifest** (`edgecv/models/manifests/<name>.yaml`) is the single source of truth for
input/output names, shapes, and artifact paths — the same manifest the runtime backend
uses, so the converter and the tracker can never disagree on I/O.

- `tools/convert.py` — CLI dispatcher.
- `tools/convert_lib/registry.py` — the `Adapter` registry.
- `tools/convert_lib/harness.py` — export + `onnx.checker` + parity.
- `tools/convert_lib/rknn.py` — generic ONNX → RKNN (deferred `rknn-toolkit2` import).
- `tools/convert_lib/adapters/<name>.py` — one per model.

The dispatcher loads the manifest, looks up the adapter, builds the module, and the harness
exports to `resolve_artifact_path(manifest.artifacts.onnx.path)`. With `--rknn` it chains
stage 3 to the manifest's rknn artifact path. The preprocessing contract (RGB/`[0,255]`/BGR
for SiamFC, etc.) lives in the manifest and is consumed by the tracker, not the converter.

## Adding a new tracker

1. Add a manifest at `edgecv/models/manifests/<name>.yaml` with the model's `io`
   (input/output names, shapes, dtypes) and `artifacts` (onnx + rknn paths). **Input order
   in `io.inputs` MUST match your module's `forward()` argument order** — the harness feeds
   inputs positionally to torch and by name to onnxruntime.
2. Add `tools/convert_lib/adapters/<name>.py`:

   ```python
   from convert_lib.registry import Adapter, register

   def build(checkpoint: str):
       # instantiate the architecture, load_state_dict(strict=True), .eval()
       return module

   register(Adapter(name="<name>", build=build))   # dynamic_axes=... if variable dims
   ```

   Vendor the architecture (as `adapters/siamfc.py` does) or import it from an installed
   package. `strict=True` makes a key/shape mismatch fail loudly.
3. Import it in `tools/convert_lib/adapters/__init__.py` so it self-registers.
4. Run: `python tools/convert.py --model <name> --checkpoint <pth>`

### Variant: upstream already exports ONNX (e.g. YOLO / ultralytics)

Some model families ship their own exporter, so you don't need a torch `nn.Module` adapter
at all — export the ONNX with the upstream tool, then convert straight to RKNN:

```bash
yolo export model=yolo.pt format=onnx        # ultralytics writes yolo.onnx
python tools/onnx_to_rknn.py --onnx yolo.onnx --out models/yolo_generic.rk3588.rknn \
    --target rk3588 --calibration-dir calib/ --inputs images
```

(Folding this behind `tools/convert.py` — an adapter that shells out to the upstream
exporter in place of `build()` and skips the torch export — is a possible future addition,
not implemented yet.)

## Notes

- `tools/` is not an installed package; `tools/convert.py` and `tools/onnx_to_rknn.py`
  insert their own directory onto `sys.path` so `import convert_lib` works when run as
  scripts. Tests do the same via `tests/conftest.py`.
- Dynamic dims: a `-1` in a manifest input shape is exported with a nominal size; declare
  the axis in the adapter's `dynamic_axes` to keep it dynamic in the ONNX graph.
