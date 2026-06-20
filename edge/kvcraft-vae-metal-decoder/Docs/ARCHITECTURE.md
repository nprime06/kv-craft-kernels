# Architecture Notes

## Upstream Shape

The model code is in `kvcraft-wm/kvcraft/src/models/wan_vae.py`; `kvcraft-wm/kvcraft-engine` only prepares the Minecraft datasets. `get_vae_model()` configures:

```python
dim=96
z_dim=16
dim_mult=[1, 2, 4, 4]
num_res_blocks=2
attn_scales=[]
temperal_downsample=[False, True, True]
```

The decoder reverses the temporal downsampling order, so it has one spatial-only upsample followed by two temporal+spatial upsample stages.

## Streaming Contract

KV Craft decode logic initializes an empty decoder feature cache. For a latent sequence with `L` latent frames:

```text
output_frames = 1 + 4 * (L - 1)
```

That means the first streamed latent produces one image frame. Each later latent produces the next four frames.

The local runtime mirrors that behavior by preserving a two-frame cache per causal 3D convolution. The `upsample3d` resample blocks follow KV Craft' special first-call behavior: the first call seeds a zero cache and skips temporal convolution.

## Tensor Layout

All runtime tensors are NHWTC as in KV Craft/JAX:

```text
B, T, H, W, C
```

Weights are exported in JAX convolution layout:

```text
3D conv: Kt, Kh, Kw, Cin, Cout
2D conv: 1, Kh, Kw, Cin, Cout
```

The Swift runtime binds every weight tensor as a separate Metal buffer for simplicity. Later profiling should pack weights into a small number of aligned buffers.

## Layer Sequence

```text
latent scale: z * VAE_STD + VAE_MEAN
conv2
decoder.conv1
decoder.middle.0 residual
decoder.middle.1 attention
decoder.middle.2 residual
decoder.upsamples.0 residual
decoder.upsamples.1 residual
decoder.upsamples.2 residual
decoder.upsamples.3 upsample2d
decoder.upsamples.4 residual
decoder.upsamples.5 residual
decoder.upsamples.6 residual
decoder.upsamples.7 upsample3d
decoder.upsamples.8 residual
decoder.upsamples.9 residual
decoder.upsamples.10 residual
decoder.upsamples.11 upsample3d
decoder.upsamples.12 residual
decoder.upsamples.13 residual
decoder.upsamples.14 residual
decoder.head.0 RMSNorm+SiLU
decoder.head.2 conv
```

## Optimized Kernel Paths

- Full steady-state MPSGraph decoder graph: default path after all causal caches are valid. It emits the four-frame steady chunk and all updated cache tensors from one graph execution.
- MPSGraph 2D-lowered causal convolution: default fast path for cached KV Craft causal convolutions and no-padding pointwise convs. Cached `Kt=3` convolutions are split into temporal slices and run through optimized MPSGraph `conv2D` kernels.
- MPSGraph `conv3D`: fallback for first-frame padded temporal convolutions and for comparison behind `SOLARIS_DISABLE_CONV3D_AS_2D=1`.
- MPSGraph `resizeNearest -> conv2D`: default fast path for spatial upsample blocks.
- Experimental phase-folded spatial upsample behind `SOLARIS_PHASE_UPSAMPLE=1`: folds nearest-upsample+3x3 conv into four low-res 2x2 phase convolutions plus a Metal interleave. It was neutral in the full decoder benchmark, so it is not default.
- MPSGraph SDPA: default exact fast path for the single middle spatial attention block.
- `scale_latent4`: four-element vectorized latent de-normalization.
- `rmsnorm_silu_tg`: one vector per threadgroup, parallel channel reduction and parallel output write.
- `add_tensors4`: four-element vectorized residual add.
- `split_channel_to_time2`: temporal pixel shuffle for 3D upsample stages.
- Fallback kernels remain available behind `SOLARIS_DISABLE_MPSGRAPH=1`: tiled `1x1`, tiled causal 3D convolution, fused upsample+conv, and tiled online-softmax attention.

## Profiling Priority

1. Wire GPU-resident display output: convert decoded RGB to BGRA texture and upscale in a render pass or MetalFX without CPU readback.
2. Validate output numerics against KV Craft/JAX on real `vae.pt` weights for both default and smaller latent grids.
3. Profile high-resolution residual blocks after the steady-state graph change. The remaining work is still dominated by `180x320x192` and `360x640x96` convs at the default output size.
4. Treat native MPS int8/int4 as not useful for this workload unless Apple exposes a quantized convolution or a faster ML pipeline path. The current MPS affine int8/int4 matrix kernels are slower than fp16 on representative tiles.
5. Keep decoded frames on GPU and render directly from a Metal texture array.

## Current Throughput Readout

The old generic MPSGraph `conv3D` path measured about `1.1 FPS` on a synthetic zero-weight archive. The earlier per-op 2D-lowered cached path measured about `3.2-3.5 FPS`.

The current default full steady-state graph path measures about `4.9-5.6 FPS` at the default `45x80 -> 360x640` output size:

```text
iterations: 8, warmup: 2
decoded frames: 29, wall time: 5.222 s
fps: 5.55
decode ms: mean 652.75, p50 674.18, p90 697.85
```

A longer sequential run measured `4.89 FPS` with p50 `796.52 ms`, so expect meaningful run-to-run variance from scheduling and thermals.

The clean path to `10-15 FPS` is smaller server-side latent grids plus local upscale:

| latent grid | decoded RGB | measured FPS | stock generator fit |
| --- | --- | ---: | --- |
| `30x53` | `240x424` | `10.79` | odd width; needs padding/crop or generator change |
| `28x50` | `224x400` | `12.23` | yes |
| `27x48` | `216x384` | `13.11` | odd height; needs padding/crop or generator change |
| `26x46` | `208x368` | `14.22` | yes |

The VAE decoder is convolutional and can run odd latent grids. The stock KV Craft generator is more constrained: it uses spatial `2x2` latent patches, RoPE from runtime grid sizes, KV-cache sizes from runtime latent H/W, and an action-module assertion for the default `880` or `1760` spatial token count. Smaller even grids should be a runtime/config/code-path change, not a new transformer architecture, but the action-module token-count assumption must be parameterized.

Bypassing attention with `SOLARIS_SKIP_ATTENTION=1` is still essentially unchanged, so attention is not the current wall. The steady-state four-frame chunk remains dominated by the high-resolution residual blocks at `180x320x192` and `360x640x96`.
