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

## Status

Foundation only — no concrete trackers yet. See
`docs/superpowers/specs/2026-05-31-edgecv-foundation-design.md`.
