#!/usr/bin/env python3
"""Create a zero-filled KVCraft VAE decoder archive for kernel benchmarking."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Iterable


def numel(shape: Iterable[int]) -> int:
    return math.prod(shape)


class Writer:
    def __init__(self, out: pathlib.Path):
        self.out = out
        self.blob = bytearray()
        self.entries: list[dict] = []

    def add(self, name: str, shape: list[int]) -> None:
        byte_count = numel(shape) * 2
        offset = len(self.blob)
        self.blob.extend(b"\x00" * byte_count)
        self.entries.append(
            {
                "name": name,
                "shape": shape,
                "dtype": "float16",
                "file": "weights.f16.bin",
                "offset": offset,
                "byteCount": byte_count,
            }
        )

    def write(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "weights.f16.bin").write_bytes(self.blob)
        manifest = {
            "format": "kvcraft-vae-decoder-f16-v1",
            "source": "synthetic zero weights for kernel benchmark",
            "tensors": self.entries,
        }
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def conv3(w: Writer, name: str, cin: int, cout: int, kt: int = 3, kh: int = 3, kw: int = 3) -> None:
    w.add(f"{name}.kernel", [kt, kh, kw, cin, cout])


def conv2(w: Writer, name: str, cin: int, cout: int, kh: int = 3, kw: int = 3) -> None:
    w.add(f"{name}.kernel", [1, kh, kw, cin, cout])


def norm(w: Writer, name: str, c: int) -> None:
    w.add(f"{name}.gamma", [c])


def residual(w: Writer, base: str, cin: int, cout: int) -> None:
    norm(w, f"{base}.residual.0", cin)
    conv3(w, f"{base}.residual.2", cin, cout)
    norm(w, f"{base}.residual.3", cout)
    conv3(w, f"{base}.residual.6", cout, cout)
    if cin != cout:
        conv3(w, f"{base}.shortcut", cin, cout, kt=1, kh=1, kw=1)


def attention(w: Writer, base: str, c: int) -> None:
    norm(w, f"{base}.norm", c)
    conv3(w, f"{base}.to_qkv", c, c * 3, kt=1, kh=1, kw=1)
    conv3(w, f"{base}.proj", c, c, kt=1, kh=1, kw=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    w = Writer(args.out)
    w.add("vae_scale.mean", [16])
    w.add("vae_scale.std", [16])
    conv3(w, "conv2", 16, 16, kt=1, kh=1, kw=1)

    conv3(w, "decoder.conv1", 16, 384)
    residual(w, "decoder.middle.0", 384, 384)
    attention(w, "decoder.middle.1", 384)
    residual(w, "decoder.middle.2", 384, 384)

    residual(w, "decoder.upsamples.0", 384, 384)
    residual(w, "decoder.upsamples.1", 384, 384)
    residual(w, "decoder.upsamples.2", 384, 384)
    conv2(w, "decoder.upsamples.3.conv", 384, 192)

    residual(w, "decoder.upsamples.4", 192, 384)
    residual(w, "decoder.upsamples.5", 384, 384)
    residual(w, "decoder.upsamples.6", 384, 384)
    conv3(w, "decoder.upsamples.7.time_conv", 384, 768, kt=3, kh=1, kw=1)
    conv2(w, "decoder.upsamples.7.conv", 384, 192)

    residual(w, "decoder.upsamples.8", 192, 192)
    residual(w, "decoder.upsamples.9", 192, 192)
    residual(w, "decoder.upsamples.10", 192, 192)
    conv3(w, "decoder.upsamples.11.time_conv", 192, 384, kt=3, kh=1, kw=1)
    conv2(w, "decoder.upsamples.11.conv", 192, 96)

    residual(w, "decoder.upsamples.12", 96, 96)
    residual(w, "decoder.upsamples.13", 96, 96)
    residual(w, "decoder.upsamples.14", 96, 96)
    norm(w, "decoder.head.0", 96)
    conv3(w, "decoder.head.2", 96, 3)

    w.write()
    print(f"wrote {len(w.entries)} tensors, {len(w.blob) / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
