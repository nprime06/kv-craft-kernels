# Edge Clients

This directory contains client-side runtimes that sit outside the B300/H100 server kernel loop.

## KV Craft VAE Metal Decoder

`edge/solaris-vae-metal-decoder/` is a native macOS Swift/Metal package for decoding streamed KV Craft VAE latents locally on an Apple GPU.

It is intended for the split-serving setup where:

```text
server: KV Craft latent generation
client: VAE latent -> RGB decode + display upscale
```

The package currently includes:

- a UDP/file/benchmark CLI
- a Metal/MPSGraph decoder runtime
- causal decoder cache handling for streamed latents
- a full steady-state MPSGraph path for warm-cache decode chunks
- a weight exporter from KV Craft/JAX/Orbax VAE weights
- low-bit MPS matmul probes
- benchmark and optimization notes

Start with:

- `edge/solaris-vae-metal-decoder/README.md`
- `edge/solaris-vae-metal-decoder/Docs/EDGE_INFERENCE_OPTIMIZATIONS.md`
- `knowledge/episodes/2026-06-20-mac-edge-vae-metal-decoder.md`

Build on macOS:

```bash
cd edge/solaris-vae-metal-decoder
swift build -c release
```

