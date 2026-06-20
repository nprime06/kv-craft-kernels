# Validation

Completed here:

- Cloned and inspected `kvcraft-wm/kvcraft-engine` at `430f56f`.
- Cloned and inspected `kvcraft-wm/kvcraft` at `68e0ed3`.
- Verified the KV Craft VAE config and decode flow from `src/models/model_loaders.py` and `src/models/wan_vae.py`.
- Verified Hugging Face model metadata includes an Orbax directory checkpoint at `vae.pt`.
- Ran `python3 -m py_compile Tools/export_kvcraft_vae_decoder.py`.
- Added optimized Metal entry points for tiled `1x1`, `co4` 3D/upsample convolutions, threadgroup RMSNorm, vectorized elementwise ops, and tiled online-softmax attention.
- Added MPSGraph fast paths for 3D convolution, resize+2D convolution, and exact SDPA attention.
- Added default cached-conv decomposition from causal 3D convolution to temporal slices of MPSGraph 2D convolution.
- Added default full steady-state MPSGraph decoder path for streamed latents after causal caches are warm.
- Added `quant-gemm-probe` for native MPS int8/int4 GEMM timing.
- Added experimental phase-folded nearest-upsample path behind `SOLARIS_PHASE_UPSAMPLE=1`.
- Built successfully with Command Line Tools 26.5 / Swift 6.3.2:

```bash
swift build -c release
```

- Benchmarked the synthetic zero-weight archive with the current default full steady-state graph path at `45x80 -> 360x640`:

```text
iterations: 8, warmup: 2
decoded frames: 29, wall time: 5.222 s
fps: 5.55
decode ms: mean 652.75, p50 674.18, p90 697.85
```

A longer sequential run measured:

```text
iterations: 20, warmup: 2
decoded frames: 77, wall time: 15.757 s
fps: 4.89
decode ms: mean 787.83, p50 796.52, p90 808.21
```

- Benchmarked smaller latent grids with the default full steady-state graph path:

```text
latent 40x71 -> RGB 320x568: fps 6.09, p50 628.92 ms
latent 36x64 -> RGB 288x512: fps 7.46, p50 513.38 ms
latent 32x57 -> RGB 256x456: fps 9.38, p50 407.31 ms
latent 30x53 -> RGB 240x424: fps 10.79, p50 355.64 ms
latent 28x50 -> RGB 224x400: fps 12.23, p50 313.47 ms
latent 27x48 -> RGB 216x384: fps 13.11, p50 293.05 ms
latent 26x46 -> RGB 208x368: fps 14.22, p50 269.81 ms
```

Odd latent dimensions are valid decoder benchmarks, but the stock KV Craft generator patchifies latents with spatial `2x2` patches. For generator-side serving without padding/cropping or architecture changes, prioritize even grids such as `28x50` and `26x46`.

- Benchmarked the old generic MPSGraph 3D convolution path:

```text
SOLARIS_DISABLE_CONV3D_AS_2D=1
iterations: 4, warmup: 1
decoded frames: 13, wall time: 11.560 s
fps: 1.12
decode ms: mean 2889.97, p50 3301.07, p90 3301.07
```

- Benchmarked the current path with attention bypassed for profiling only:

```text
SOLARIS_SKIP_ATTENTION=1
iterations: 8, warmup: 2
decoded frames: 29, wall time: 5.153 s
fps: 5.63
decode ms: p50 667.52
```

- Benchmarked the experimental phase upsample path with the full steady-state graph:

```text
SOLARIS_PHASE_UPSAMPLE=1
iterations: 8, warmup: 2
decoded frames: 29, wall time: 5.150 s
fps: 5.63
decode ms: p50 671.90
```

This is within noise of the default path and is not enabled by default.

- Benchmarked native MPSGraph 3D convolution inside the full steady-state graph:

```text
SOLARIS_STEADY_NATIVE_CONV3D=1
iterations: 8, warmup: 2
decoded frames: 29, wall time: 21.946 s
fps: 1.32
decode ms: p50 3061.24
```

This is much slower than the default 2D-lowered graph path.

- Benchmarked native MPS low-bit matmul on representative conv-equivalent tiles:

```text
case: mid_res_conv_tile_4096 M=4096 K=5184 N=192
fp16_mps_ndarray p50: 1.427 ms
int8_affine_both p50: 5.180 ms
int4_affine_both p50: 7.016 ms
```

- Ran fallback handwritten-kernel comparison:

```text
SOLARIS_DISABLE_MPSGRAPH=1
iterations: 1, warmup: 0
decoded frames: 1, wall time: 7.706 s
fps: 0.13
```

Still open:

- `xcrun metal` is unavailable in the selected Command Line Tools, so offline Metal compilation is not available. Runtime source compilation works.
- The runtime has not yet been compared numerically against JAX on real KV Craft `vae.pt` weights.
- The current benchmark uses zero weights, so it is a kernel/dispatch throughput test, not a quality validation.
- There is no active XCTest target. `swift test -c release` exits with "no tests found"; use `swift build -c release` plus the benchmark command as the current validation path.
- Native int8/int4 MPS matmul was slower than fp16 in the probe; no full-decoder quantized path is enabled.
- The display path is not wired yet. The decoder can produce smaller RGB tensors, and the Metal shader includes RGB-to-BGRA texture conversion, but the CLI currently logs timing rather than presenting/upscaling frames.

Next validation:

```bash
swift build -c release
.build/release/kvcraft-vae-metal --weights /tmp/kvcraft-vae-dummy --benchmark 8 --warmup 2
.build/release/kvcraft-vae-metal --weights /tmp/kvcraft-vae-dummy --benchmark 12 --warmup 2 --latent-height 28 --latent-width 50
```

Then export weights and compare one streamed decode chunk against KV Craft/JAX with a fixed latent tensor.
