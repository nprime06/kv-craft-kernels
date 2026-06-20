# KV Craft VAE Metal Decoder

This is a native macOS Swift/Metal serving scaffold for the KV Craft VAE decoder. It targets the current KV Craft model repo layout inspected at commit `68e0ed3` and the KV Craft Engine data repo inspected at commit `430f56f`.

The linked `kvcraft-wm/kvcraft-engine` repo is a data-collection engine. The VAE implementation lives in `kvcraft-wm/kvcraft`, specifically `src/models/wan_vae.py`.

## What Is Implemented

- Static KV Craft WanVAE decoder graph:
  - `dim=96`
  - `z_dim=16`
  - `dim_mult=[1, 2, 4, 4]`
  - spatial decode factor `8`
  - temporal stream decode: first latent emits 1 frame, later latents emit 4 frames
- Custom Metal kernels for:
  - RMSNorm plus optional SiLU
  - residual add
  - temporal channel-to-time split for 3D upsampling
  - causal-source packing for MPSGraph 3D convolutions
  - QKV packing for MPSGraph scaled-dot-product attention
  - fallback tiled convolution and tiled spatial attention paths
  - optional RGB tensor to BGRA8 texture conversion kernel
- MPSGraph fast paths for:
  - full steady-state decoder graph for streamed latents after caches are warm
  - cached causal `conv3D` decomposed into temporal slices of optimized `conv2D`
  - fallback `conv3D` in NDHWC/DHWIO layout for first-frame padded temporal convs
  - nearest-neighbor resize plus `conv2D` for spatial upsample blocks
  - exact single-head scaled-dot-product spatial attention
- UDP serving CLI for streamed latent datagrams.
- Runtime-selectable latent grid size with `--latent-height` and `--latent-width`.
- Python exporter from Hugging Face/Orbax `vae.pt` to a simple `float16` archive.
- `quant-gemm-probe` executable for testing Apple MPS int8/int4 matrix kernels on conv-equivalent tiles.

## Build

```bash
swift build -c release
```

The package now requires macOS 15+ because it uses MPSGraph SDPA. Full Xcode is not required for this path; Command Line Tools 26.5 with Swift 6.3.2 built it successfully. `xcrun metal` is still not required because the runtime compiles `KV CraftVAE.metal` from package resources with `device.makeLibrary(source:options:)`.

## Export Weights

From a KV Craft Python environment:

```bash
cd /path/to/kvcraft
hf download nyu-visionx/kvcraft --local-dir ./pretrained
python /path/to/kvcraft-vae-metal-decoder/Tools/export_kvcraft_vae_decoder.py \
  --kvcraft-root . \
  --vae-checkpoint ./pretrained/vae.pt \
  --out ./kvcraft-vae-decoder-f16
```

The exporter converts JAX `bfloat16` checkpoint tensors to IEEE `float16`, because Apple Metal does not expose `bfloat16` arithmetic.

## Serve Streamed Latents

The CLI expects one latent per UDP datagram:

```text
B=1, T=1, H=<latent-height>, W=<latent-width>, C=16, little-endian float16
```

The decoder CLI default latent grid is `45x80`, which decodes to `360x640` RGB. Output size is always `latent H/W * 8`. Smaller latent grids can be streamed when the server-side producer is configured for the same grid, then the decoded RGB can be upscaled for display.

For the stock KV Craft world-model generator, prefer even latent heights and widths. The generator patchifies latents with spatial `2x2` patches, and the action module currently has default-resolution token-count assumptions. The decoder itself can benchmark odd grids, but serving them from the unmodified generator would require padding/cropping or generator-side code changes.

For the repo-level split-serving bridge, use `28x50` by default. It fits in one safe UDP datagram after the bridge strips the framing header and forwards the raw latent payload locally. A full `45x80` latent is `115200` bytes and needs fragmentation or a stream transport; do not send it as one UDP packet over the current bridge.

Run:

```bash
.build/release/kvcraft-vae-metal \
  --weights /path/to/kvcraft-vae-decoder-f16 \
  --udp-port 7777
```

For example, a `28x50` latent grid decodes to `224x400` and measured about `12 FPS` on the synthetic benchmark:

```bash
.build/release/kvcraft-vae-metal \
  --weights /path/to/kvcraft-vae-decoder-f16 \
  --udp-port 7777 \
  --latent-height 28 \
  --latent-width 50
```

For a single raw latent file:

```bash
.build/release/kvcraft-vae-metal \
  --weights /path/to/kvcraft-vae-decoder-f16 \
  --latent latent.f16
```

## Benchmark

Create the synthetic zero-weight archive:

```bash
python3 Tools/make_dummy_archive.py --out /tmp/kvcraft-vae-dummy
```

Run:

```bash
.build/release/kvcraft-vae-metal \
  --weights /tmp/kvcraft-vae-dummy \
  --benchmark 8 \
  --warmup 2
```

Current measured result on this Mac with the default full steady-state graph path at the default `45x80 -> 360x640` size:

```text
iterations: 8, warmup: 2
decoded frames: 29, wall time: 5.222 s
fps: 5.55
decode ms: mean 652.75, p50 674.18, p90 697.85
```

A longer sequential run measured `4.89 FPS` with p50 `796.52 ms`; expect some run-to-run variance from GPU scheduling and thermals.

Smaller-grid sequential sweep with the same synthetic archive:

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

The recommended first target for a 10-15 FPS display path is `28x50 -> 224x400`, upscaled to the window with Metal texture sampling or MetalFX. Use `26x46 -> 208x368` if you want more FPS headroom.

The old generic MPSGraph `conv3D` path is still available for comparison:

```text
SOLARIS_DISABLE_CONV3D_AS_2D=1
iterations: 4, warmup: 1
decoded frames: 13, wall time: 11.560 s
fps: 1.12
decode ms: mean 2889.97, p50 3301.07
```

Comparison switches:

```bash
SOLARIS_DISABLE_STEADY_GRAPH=1 ...  # old per-op cached MPSGraph path
SOLARIS_DISABLE_CONV3D_AS_2D=1 ...  # old generic MPSGraph conv3D path
SOLARIS_PHASE_UPSAMPLE=1 ...        # experimental low-res phase upsample rewrite; neutral in full benchmark
SOLARIS_DISABLE_MPSGRAPH=1 ...      # fallback handwritten Metal conv/attention path
SOLARIS_SKIP_ATTENTION=1 ...        # profiling only; bypasses attention and breaks quality
```

`SOLARIS_SKIP_ATTENTION=1` produces essentially the same result as the default path, so the remaining wall is the convolutional decoder stack, not the middle attention block.

To probe native low-bit MPS matmul paths:

```bash
.build/release/quant-gemm-probe --iterations 10 --warmup 3
```

On this M4 Pro, MPS affine int8/int4 matmul did not speed up conv-shaped tiles. For example, `M=4096,K=5184,N=192` measured fp16 p50 `1.427 ms`, int8-both p50 `5.180 ms`, and int4-both p50 `7.016 ms`.

## Current Limitations

This is not yet a numerically verified production runtime. It is the low-level Metal implementation scaffold needed to start profiling and iterating locally.

The biggest limiting factor is the exact convolutional workload. A steady-state latent chunk emits four frames and is roughly `2.82` trillion MACs at the default `360x640` output before counting elementwise ops and dispatch overhead. Reaching `10 FPS` at this full resolution still requires a much faster Apple ML/Metal compiled convolution path or a model/runtime contract change.

The most practical contract change found so far is reducing the streamed latent grid and upscaling for display. This only helps if the server computes and sends smaller latents; decoding a full `45x80` latent and resizing after decode does not reduce decoder cost.

Other limits:

- Metal lacks `bfloat16`, so the current archive is `float16`; exact JAX parity will need tolerance checks.
- Apple public Metal/MPSGraph APIs do not expose a simple FP8/MXFP4 conv path here. The public int8/int4 MPS matmul kernels tested slower than fp16 for these conv-shaped tiles, and MPSGraph does not expose a quantized conv op in this SDK.
- The default path now uses a full steady-state MPSGraph after the causal caches are warm. The first latent still uses the op-by-op path to seed caches.
- The CLI decodes to a GPU tensor and logs timing; display integration should consume the output buffer or BGRA texture without CPU readback, then upscale in a render pass.
- The weight exporter requires the KV Craft Python/JAX/Orbax environment and the full `vae.pt` checkpoint directory from Hugging Face.
