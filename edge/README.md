# Edge Clients

This directory contains client-side runtimes that sit outside the B300/H100 server kernel loop.

## KV Craft VAE Metal Decoder

`edge/kvcraft-vae-metal-decoder/` is a native macOS Swift/Metal package for decoding streamed KV Craft VAE latents locally on an Apple GPU.

It is intended for the split-serving setup where:

```text
server: KV Craft latent generation
laptop bridge: action upload + latent receive
client decoder: VAE latent -> RGB decode + display upscale
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

- `edge/kvcraft-vae-metal-decoder/README.md`
- `edge/kvcraft-vae-metal-decoder/Docs/EDGE_INFERENCE_OPTIMIZATIONS.md`
- `knowledge/episodes/2026-06-20-mac-edge-vae-metal-decoder.md`

Build on macOS:

```bash
cd edge/kvcraft-vae-metal-decoder
swift build -c release
```

## Split-Serving Flow

Run the decoder on the laptop:

```bash
cd edge/kvcraft-vae-metal-decoder
.build/release/kvcraft-vae-metal --weights /path/to/kvcraft-vae-decoder-f16 \
  --udp-port 7777 --latent-height 28 --latent-width 50
```

Run the laptop bridge in another terminal:

```bash
oasis-forge laptop-bridge --server-host <central-host> --player-id 0
```

The bridge sends actions to the central server and forwards received raw latent payloads to the
local decoder. Local input capture can feed the bridge with JSON datagrams on `127.0.0.1:7790`.
