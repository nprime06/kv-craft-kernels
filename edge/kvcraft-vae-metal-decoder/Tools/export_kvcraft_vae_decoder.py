#!/usr/bin/env python3
"""
Export KVCraft' WanVAE decoder weights from the Hugging Face Orbax checkpoint
into the simple float16 archive consumed by the Swift/Metal runtime.

Run this from an environment that can import the KVCraft repo:

  cd /path/to/kvcraft
  python /path/to/export_kvcraft_vae_decoder.py \
    --kvcraft-root . \
    --vae-checkpoint ./pretrained/vae.pt \
    --out ./kvcraft-vae-decoder-f16
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kvcraft-root", required=True, type=pathlib.Path)
    parser.add_argument("--vae-checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    return parser.parse_args()


def value(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def to_f16_bytes(x: Any) -> tuple[list[int], bytes]:
    arr32 = np.asarray(value(x), dtype=np.float32)
    arr16 = arr32.astype("<f2", copy=False)
    return list(arr16.shape), arr16.tobytes(order="C")


class ArchiveWriter:
    def __init__(self, out_dir: pathlib.Path, source: str):
        self.out_dir = out_dir
        self.source = source
        self.entries: list[dict[str, Any]] = []
        self.blob = bytearray()

    def add(self, name: str, x: Any, reshape_conv2d: bool = False) -> None:
        arr = value(x)
        if reshape_conv2d:
            arr = np.asarray(arr)
            if arr.ndim == 4:
                arr = arr[None, ...]
        shape, payload = to_f16_bytes(arr)
        offset = len(self.blob)
        self.blob.extend(payload)
        self.entries.append(
            {
                "name": name,
                "shape": shape,
                "dtype": "float16",
                "file": "weights.f16.bin",
                "offset": offset,
                "byteCount": len(payload),
            }
        )

    def write(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "weights.f16.bin").write_bytes(self.blob)
        manifest = {
            "format": "kvcraft-vae-decoder-f16-v1",
            "source": self.source,
            "tensors": self.entries,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def emit_conv(writer: ArchiveWriter, base: str, conv: Any) -> None:
    writer.add(f"{base}.kernel", conv.kernel, reshape_conv2d=True)
    bias = getattr(conv, "bias", None)
    if bias is not None:
        writer.add(f"{base}.bias", bias)


def emit_norm(writer: ArchiveWriter, base: str, norm: Any) -> None:
    writer.add(f"{base}.gamma", norm.gamma)
    bias = getattr(norm, "bias", 0.0)
    if not isinstance(bias, float):
        writer.add(f"{base}.bias", bias)


def export(args: argparse.Namespace) -> None:
    root = args.kvcraft_root.resolve()
    sys.path.insert(0, str(root))

    import orbax.checkpoint as ocp
    from flax import nnx
    from src.models.model_loaders import get_vae_model
    from src.models.wan_vae import (
        VAE_MEAN,
        VAE_STD,
        AttentionBlock,
        ResidualBlock,
        Resample,
    )

    vae = get_vae_model()
    graph, state = nnx.split(vae)
    restored = ocp.StandardCheckpointer().restore(str(args.vae_checkpoint), state)
    vae = nnx.merge(graph, restored)

    writer = ArchiveWriter(
        args.out,
        source=f"{args.vae_checkpoint} via {root}",
    )
    writer.add("vae_scale.mean", VAE_MEAN)
    writer.add("vae_scale.std", VAE_STD)

    emit_conv(writer, "conv2", vae.conv2)

    decoder = vae.decoder
    emit_conv(writer, "decoder.conv1", decoder.conv1)

    for i, layer in enumerate(decoder.middle):
        base = f"decoder.middle.{i}"
        if isinstance(layer, ResidualBlock):
            emit_residual(writer, base, layer)
        elif isinstance(layer, AttentionBlock):
            emit_attention(writer, base, layer)

    for i, layer in enumerate(decoder.upsamples):
        base = f"decoder.upsamples.{i}"
        if isinstance(layer, ResidualBlock):
            emit_residual(writer, base, layer)
        elif isinstance(layer, Resample):
            emit_resample(writer, base, layer)

    emit_norm(writer, "decoder.head.0", decoder.head[0])
    emit_conv(writer, "decoder.head.2", decoder.head[2])
    writer.write()
    print(f"wrote {len(writer.entries)} tensors to {args.out}")


def emit_residual(writer: ArchiveWriter, base: str, block: Any) -> None:
    emit_norm(writer, f"{base}.residual.0", block.residual[0])
    emit_conv(writer, f"{base}.residual.2", block.residual[2])
    emit_norm(writer, f"{base}.residual.3", block.residual[3])
    emit_conv(writer, f"{base}.residual.6", block.residual[6])
    if block.shortcut is not None:
        emit_conv(writer, f"{base}.shortcut", block.shortcut)


def emit_attention(writer: ArchiveWriter, base: str, block: Any) -> None:
    emit_norm(writer, f"{base}.norm", block.norm)
    emit_conv(writer, f"{base}.to_qkv", block.to_qkv)
    emit_conv(writer, f"{base}.proj", block.proj)


def emit_resample(writer: ArchiveWriter, base: str, block: Any) -> None:
    if hasattr(block, "time_conv"):
        emit_conv(writer, f"{base}.time_conv", block.time_conv)
    if hasattr(block, "conv"):
        emit_conv(writer, f"{base}.conv", block.conv)


if __name__ == "__main__":
    export(parse_args())
