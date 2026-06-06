"""ONNX -> RKNN conversion (ARCHITECTURE.md §11). Host-only, x86. SCAFFOLD: the
rknn-toolkit2 import is deferred so this module imports anywhere; conversion only
runs where the toolkit is installed. Not exercised in CI.

rknn-toolkit2 is not on PyPI cleanly — install it on an x86 host from Rockchip's
release wheels. INT8 quantisation needs a folder of representative calibration
images (frames resembling deployment input).

Usage (on a host with rknn-toolkit2):
    python tools/onnx_to_rknn.py \
        --onnx models/siamfc_generic.onnx \
        --out models/siamfc_generic.rk3588.rknn \
        --target rk3588 \
        --calibration-dir calib/ \
        --inputs exemplar search
"""

from __future__ import annotations

import argparse

_INSTALL_HINT = (
    "rknn-toolkit2 is not importable. Install it on an x86 host from Rockchip's "
    "release wheels (it is not on PyPI). This tool runs offline; the device only "
    "runs the lite runtime (ARCHITECTURE.md §11, §12)."
)


def _import_rknn():
    from rknn.api import RKNN  # type: ignore

    return RKNN


def _write_dataset_file(calibration_dir: str) -> str:
    """RKNN's build() wants a text file listing one calibration image per line."""
    from pathlib import Path

    imgs = sorted(
        str(p) for p in Path(calibration_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not imgs:
        raise SystemExit(f"no calibration images found in {calibration_dir!r}")
    listing = Path(calibration_dir) / "dataset.txt"
    listing.write_text("\n".join(imgs) + "\n")
    return str(listing)


def convert(onnx_path: str, out_path: str, target: str,
            calibration_dir: str | None, input_names: list[str]) -> str:
    try:
        RKNN = _import_rknn()
    except Exception as e:  # pragma: no cover - depends on host toolkit
        raise RuntimeError(_INSTALL_HINT) from e

    quantize = calibration_dir is not None
    rknn = RKNN(verbose=True)
    # mean/std [0,255] passthrough: these weights consume raw pixels (scale handled
    # in the tracker preprocessing, not the model). Adjust if a future model normalises.
    rknn.config(mean_values=[[0, 0, 0]] * len(input_names),
                std_values=[[1, 1, 1]] * len(input_names),
                target_platform=target)
    if rknn.load_onnx(model=onnx_path, inputs=input_names) != 0:
        raise RuntimeError(f"load_onnx failed for {onnx_path!r}")
    dataset = _write_dataset_file(calibration_dir) if quantize else None
    if rknn.build(do_quantization=quantize, dataset=dataset) != 0:
        raise RuntimeError("rknn build failed")
    if rknn.export_rknn(out_path) != 0:
        raise RuntimeError(f"export_rknn failed for {out_path!r}")
    rknn.release()
    print(f"exported {out_path} (target={target}, quantized={quantize})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX -> RKNN (host-only scaffold)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--calibration-dir", default=None,
                    help="folder of representative images; enables INT8 quantisation")
    ap.add_argument("--inputs", nargs="+", default=["exemplar", "search"],
                    help="model input names (order must match the ONNX graph)")
    args = ap.parse_args()
    convert(args.onnx, args.out, args.target, args.calibration_dir, args.inputs)


if __name__ == "__main__":
    main()
