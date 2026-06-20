# 2026-06-20 - Mac edge VAE decoder path for KV Craft

Added a native macOS Swift/Metal KV Craft VAE decoder under:

```text
edge/kvcraft-vae-metal-decoder/
```

This is separate from the B300/H100 server-kernel loop. It targets the client-side bottleneck in a split-serving design: the server streams KV Craft latents, while the Mac decodes VAE latents to RGB locally on the Apple GPU.

## What was added

- Static KV Craft WanVAE decoder runtime for `dim=96`, `z_dim=16`, `dim_mult=[1,2,4,4]`.
- UDP/file/benchmark CLI for one latent datagram per streamed step.
- Causal 3D-conv cache handling matching the KV Craft streaming decode contract.
- MPSGraph fast paths for convolutions, nearest-upsample plus conv, and scaled-dot-product attention.
- Full steady-state MPSGraph decoder path after caches are warm.
- Custom Metal kernels for elementwise glue, cache/layout transforms, fallback conv/attention, and RGB-to-BGRA conversion.
- Export tool for KV Craft/JAX/Orbax VAE weights into a local f16 archive.
- Quantization probe for native MPS int8/int4 matmul paths.
- Edge optimization writeup with benchmark data and remaining blockers.

## Throughput results on the Mac test machine

Synthetic zero-weight archive, default full steady-state graph:

| path | measured throughput |
| --- | ---: |
| handwritten fallback kernels | `0.13 FPS` cold one-frame comparison |
| generic MPSGraph `conv3D` | `1.12 FPS` |
| per-op MPSGraph with 2D-lowered cached convs | `3.2-3.5 FPS` |
| full steady-state graph, `45x80 -> 360x640` | `4.9-5.6 FPS` |
| full steady-state graph, `28x50 -> 224x400` | `12.23 FPS` |
| full steady-state graph, `26x46 -> 208x368` | `14.22 FPS` |

The current practical route to 10-15 FPS on this Mac is smaller decoded frames plus GPU-resident upscale. The clean target is `28x50` latent -> `224x400` RGB -> display upscale.

## What did not help

- Native MPSGraph `conv3D` inside the steady graph was much slower than the 2D-lowered path.
- Bypassing attention barely moved wall time, so attention is not the local decoder bottleneck.
- The phase-folded upsample experiment was neutral.
- Public MPS int8/int4 matmul kernels were slower than fp16 on representative conv-equivalent tiles.

## Integration implications

If the server can emit smaller even latent grids, use `28x50` or `26x46`. The stock KV Craft generator patchifies latents with spatial `2x2` patches, so even latent dimensions are safest.

If the server cannot be changed or retrained, the no-retrain experiments are:

- crop full latents before decode: faster but cropped/zoomed view
- resize full latents before decode: whole-frame approximation but possible latent-space artifacts

Neither experiment is guaranteed to preserve quality. They should be judged visually and against trajectory drift, not just local decode FPS.

## Still open

- Export real KV Craft VAE weights and compare fixed-latent output against JAX numerically.
- Wire decoded GPU tensor to BGRA texture presentation and upscale without CPU readback.
- Measure real UDP stream latency and jitter.
- Confirm the exact server-side latent shape in the active serving path.
