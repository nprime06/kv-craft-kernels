# Edge Inference Optimization Summary

This document summarizes the work done to make the KV Craft VAE decoder viable for local edge inference on a Mac GPU while the world-model latents are generated on a server and streamed to the client.

The target workload is only the VAE decoder, not the full KV Craft world model. The server emits latent chunks. The Mac receives those latent chunks, keeps the decoder temporal caches warm, decodes RGB frames locally, and should eventually present them directly from GPU memory.

## Starting Point

The upstream repository split matters:

- `kvcraft-wm/kvcraft-engine` is the data collection engine.
- The VAE implementation is in `kvcraft-wm/kvcraft/src/models/wan_vae.py`.

The KV Craft WanVAE decoder configuration inspected here is:

```text
dim = 96
z_dim = 16
dim_mult = [1, 2, 4, 4]
spatial decode factor = 8
temporal stream contract = first latent emits 1 frame, later latents emit 4 frames
```

The decoder default used by the local CLI is a `45x80` latent grid, which decodes to `360x640` RGB. The decoder can run other latent H/W values, with output size always equal to `latent H/W * 8`.

## Runtime Scaffold

The local project is a Swift package with a Metal/MPSGraph decoder runtime:

- `kvcraft-vae-metal`: UDP/file/benchmark CLI for streamed latents.
- `KV CraftVaeMetalDecoder`: reusable Swift library for GPU tensors, weights, kernels, and decode execution.
- `quant-gemm-probe`: standalone executable for native MPS int8/int4 matmul timing.
- `Tools/export_kvcraft_vae_decoder.py`: exporter from KV Craft/JAX/Orbax VAE weights to a simple f16 archive.

The runtime uses `float16` weights and activations because Apple Metal does not expose `bfloat16` arithmetic. The upstream checkpoint tensors are converted from JAX `bfloat16` to IEEE `float16` in the exporter.

Full Xcode is not required for the current path. The package builds with Command Line Tools 26.5 and Swift 6.3.2. Offline `xcrun metal` is not required because the runtime compiles `KV CraftVAE.metal` from package resources at startup with `device.makeLibrary(source:options:)`.

## Streaming and Cache Optimizations

KV Craft VAE decode is causal in time. A naive one-shot decode would repeatedly recompute temporal context. The local runtime mirrors the upstream cache behavior:

- Each causal 3D convolution keeps a two-frame feature cache.
- The first latent frame seeds caches and emits one RGB frame.
- Every later latent frame runs the steady-state path and emits four RGB frames.
- Cache tensors are updated on GPU after every decode.

The default steady-state path now uses a full MPSGraph execution for the four-frame decode chunk. It returns both:

- the decoded RGB tensor
- every updated causal cache tensor

That avoids launching a separate graph for every individual norm, convolution, add, and upsample once the stream is warm. Old cache tensors are retained until the command buffer completes, so cache replacement does not race GPU execution.

## Convolution Path

The main wall is convolution, not attention. The decoder has many high-resolution residual blocks after upsampling, especially around `180x320x192` and `360x640x96`.

The first MPSGraph path used generic `conv3D`. That was correct but slow on this workload:

```text
SOLARIS_DISABLE_CONV3D_AS_2D=1
iterations: 4, warmup: 1
decoded frames: 13, wall time: 11.560 s
fps: 1.12
decode ms: mean 2889.97, p50 3301.07
```

The faster path decomposes cached causal `Kt=3` 3D convolutions into temporal slices of optimized MPSGraph `conv2D`:

```text
for each temporal kernel slice:
  slice cached/current source along T
  run conv2D over H/W
sum the temporal slice outputs
add bias
reshape back to NHWTC
```

This gives MPSGraph access to Apple's optimized 2D convolution kernels, which are much faster than the generic 3D convolution path on this Mac. First-frame padded temporal convolutions still have a fallback path, but steady-state cached convolutions use the 2D-lowered path.

Native 3D convolution was also tried inside the full steady-state graph and was much slower:

```text
SOLARIS_STEADY_NATIVE_CONV3D=1
iterations: 8, warmup: 2
decoded frames: 29, wall time: 21.946 s
fps: 1.32
decode ms: p50 3061.24
```

The default remains 2D-lowered convolution.

## Graph Fusion

The largest speedup after the convolution change was moving from per-op cached MPSGraph plans to one full steady-state MPSGraph.

Earlier per-op 2D-lowered path:

```text
rough range: 3.2-3.5 FPS
```

Current full steady-state graph at default `45x80 -> 360x640`:

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

Expect run-to-run variance from GPU scheduling and thermals, but the practical full-size throughput is now roughly `5 FPS`.

## Attention Path

The decoder has one middle spatial attention block. Two paths exist:

- default MPSGraph scaled-dot-product attention
- fallback custom tiled Metal attention with online softmax

The fallback tiled attention was written to avoid explicitly materializing a full score matrix in our own Metal code. The default path uses MPSGraph SDPA, so score-matrix materialization is delegated to Apple's implementation rather than implemented in this project.

Attention is not the current bottleneck. Bypassing it for profiling was almost unchanged:

```text
SOLARIS_SKIP_ATTENTION=1
iterations: 8, warmup: 2
decoded frames: 29, wall time: 5.153 s
fps: 5.63
decode ms: p50 667.52
```

Because skipping attention breaks quality and barely changes wall time, attention optimization is not the next priority.

## Custom Metal Kernels

The project includes custom Metal kernels for glue work and fallbacks:

- latent de-normalization: `scale_latent4`
- RMSNorm plus optional SiLU: `rmsnorm_silu_tg`
- residual add: `add_tensors4`
- temporal channel-to-time split for 3D upsample stages: `split_channel_to_time2`
- causal source packing for MPSGraph convolution inputs
- QKV split/packing for MPSGraph SDPA
- phase-major depth-to-space helper for experimental upsample folding
- fallback tiled convolution kernels
- fallback tiled online-softmax attention kernels
- optional NHWTC RGB to BGRA texture conversion: `nhwtc3_to_bgra8`

The production-fast path intentionally leans on MPSGraph for convolutions and SDPA, because Apple's tuned kernels beat handwritten generic Metal kernels for this workload. The custom kernels are still important for layout transforms, cache updates, cheap elementwise work, fallback operation, and future display integration.

## Spatial Upsample Experiment

The decoder has nearest-neighbor spatial upsample followed by convolution. An experimental phase-folded path was added behind:

```bash
SOLARIS_PHASE_UPSAMPLE=1
```

It rewrites nearest-upsample plus 3x3 convolution as four low-resolution phase convolutions plus an interleave. This is a standard trick for reducing explicit high-resolution work.

In the full decoder benchmark it was neutral:

```text
SOLARIS_PHASE_UPSAMPLE=1
iterations: 8, warmup: 2
decoded frames: 29, wall time: 5.150 s
fps: 5.63
decode ms: p50 671.90
```

Because it was within noise of the default path and adds complexity, it is not enabled by default.

## Low-Bit Quantization Probe

Native MPS int8/int4 matrix multiplication was tested on conv-equivalent tiles with `quant-gemm-probe`.

Representative result:

```text
case: mid_res_conv_tile_4096 M=4096 K=5184 N=192
fp16_mps_ndarray p50: 1.427 ms
int8_affine_both p50: 5.180 ms
int4_affine_both p50: 7.016 ms
```

For this decoder and this public SDK path, int8/int4 MPS matmul was slower than fp16. There is no full-decoder quantized path enabled.

The practical conclusion is:

- FP8/MXFP4 would be interesting only if exposed through a fast Apple convolution or ML pipeline path.
- Manually lowering the decoder convolutions to public MPS int8/int4 matmuls is not currently attractive.
- Post-training quantization or quantization-aware retraining is not the immediate blocker, because the available low-bit math path did not speed up the representative kernels.

## Smaller Latent Grids

The biggest remaining lever is to reduce the decoder's spatial workload. Since the VAE decoder is convolutional, it can accept smaller latent H/W values.

Sequential synthetic benchmark sweep:

| latent grid | decoded RGB | measured FPS | stock generator fit |
| --- | --- | ---: | --- |
| `45x80` | `360x640` | `4.9-5.6` | decoder default |
| `40x71` | `320x568` | `6.09` | odd width; needs padding/crop or generator change |
| `36x64` | `288x512` | `7.46` | yes |
| `32x57` | `256x456` | `9.38` | odd width; needs padding/crop or generator change |
| `30x53` | `240x424` | `10.79` | odd width; needs padding/crop or generator change |
| `28x50` | `224x400` | `12.23` | yes |
| `27x48` | `216x384` | `13.11` | odd height; needs padding/crop or generator change |
| `26x46` | `208x368` | `14.22` | yes |

The best clean target found so far is:

```text
28x50 latent -> 224x400 RGB -> upscale for display
```

That lands in the `10-15 FPS` range while keeping even latent dimensions. The stock KV Craft world model uses spatial `2x2` latent patches, so even latent H/W is the safer no-architecture-change path.

If the server cannot generate smaller latents, there are still no-retrain experiments:

- crop full latents before decoding: fast but produces a cropped/zoomed view
- resize full latents before decoding: preserves whole-frame framing but may introduce latent-space artifacts

Neither is equivalent to retraining or native smaller-grid generation. They are practical quality experiments, not guaranteed model-preserving transformations.

## Display Path

The current CLI decodes to a GPU tensor and logs timing. It does not yet present frames in a window.

The intended edge display path is:

```text
streamed latent datagram
-> GPU latent tensor
-> VAE decoder steady-state graph
-> NHWTC RGB tensor on GPU
-> RGB-to-BGRA texture kernel
-> render pass / MetalFX upscale
-> display
```

The `nhwtc3_to_bgra8` kernel already exists, but the windowed renderer/upscaler is not wired into the CLI yet. Keeping this path GPU-resident is important; CPU readback would waste the speed gained by the decoder work.

## Throughput Timeline

The useful performance progression was:

| path | measured throughput |
| --- | ---: |
| handwritten fallback kernels | `0.13 FPS` in cold one-frame comparison |
| generic MPSGraph `conv3D` | `1.12 FPS` |
| per-op MPSGraph with 2D-lowered cached convs | `3.2-3.5 FPS` |
| full steady-state graph, default size | `4.9-5.6 FPS` |
| full steady-state graph, `28x50 -> 224x400` | `12.23 FPS` |
| full steady-state graph, `26x46 -> 208x368` | `14.22 FPS` |

These numbers were measured with a synthetic zero-weight archive. They are kernel and dispatch throughput measurements, not visual quality validation.

## Current Limiting Factors

The remaining full-resolution bottleneck is the convolutional workload. A steady-state latent chunk emits four frames and is roughly `2.82` trillion MACs at `360x640` before counting elementwise work and dispatch overhead.

At default resolution, getting to `10 FPS` likely needs one of:

- a faster Apple ML/Metal compiled convolution pipeline
- a lower spatial decode contract
- a changed decoder architecture
- a hardware class with substantially more sustained conv throughput

For the current Mac path, lower spatial decode plus GPU-resident upscale is the practical route to `10-15 FPS`.

## What Still Needs Validation

Before treating this as production quality:

- Export real KV Craft VAE weights and compare one fixed latent decode against JAX numerically.
- Test visual quality at default size and at `28x50`.
- Test no-retrain latent resize/crop from full server latents if changing the server grid is not possible.
- Wire the BGRA texture display path and confirm no CPU readback.
- Measure real streamed UDP latency and jitter, not just local synthetic decode timing.
- Confirm the exact server-side latent shape. The local CLI default uses `45x80 -> 360x640`, while the inspected upstream runner preprocesses some paths to `352x640`, which implies `44x80` latents.

