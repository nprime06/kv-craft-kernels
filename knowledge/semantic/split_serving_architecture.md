# Split-Serving Architecture

This repo now has a concrete split-serving seam:

```text
laptop input -> laptop bridge -> central B300/H100 latent generator
central latent generator -> laptop bridge -> local Metal VAE decoder -> display
```

The design keeps the expensive world-model latent generation centralized while moving VAE frame
decode to each laptop's Apple GPU.

## Data Flow

1. Each laptop captures local player actions.
2. The laptop bridge sends compact action packets to the central server.
3. The central server updates the authoritative action timeline and generates the next latent chunk.
4. The central server sends framed latent packets back to each laptop.
5. The laptop bridge validates the frame header and forwards only the raw NHWTC f16 latent payload
   to the local `kvcraft-vae-metal` decoder UDP port.
6. The local decoder keeps its causal VAE caches warm and decodes RGB frames on the Mac GPU.

## Current Wire Protocol

Implemented in `src/oasis_forge/streaming.py`.

Action packet magic:

```text
KVCA
```

Action fields:

```text
version
flags
player_id
sequence
client_time_ns
buttons bitmask
mouse_dx
mouse_dy
```

Latent packet magic:

```text
KVCL
```

Latent fields:

```text
version
flags
player_id
sequence
server_time_ns
latent_height
latent_width
channels
dtype
payload_len
payload bytes
```

The current dtype is little-endian IEEE f16 (`DTYPE_F16_LE`). The payload is exactly the raw tensor
expected by the Metal decoder:

```text
B=1, T=1, H=<latent_height>, W=<latent_width>, C=16, f16
```

## Ports

Defaults:

| path | default |
| --- | ---: |
| laptop action bridge -> central server | UDP `7780` |
| central server -> laptop latent bridge | UDP `7781` |
| local input capture -> laptop bridge | UDP `7790` |
| laptop bridge -> local Metal decoder | UDP `7777` |

## Datagram Size Constraint

A full `45x80` f16 latent is:

```text
45 * 80 * 16 * 2 = 115200 bytes
```

That does not fit in a safe single UDP datagram. The current bridge intentionally defaults to
`28x50`, which is:

```text
28 * 50 * 16 * 2 = 44800 bytes
```

That fits as one framed UDP datagram and matches the measured `~12 FPS` local decode target.

For larger latent grids, add fragmentation/reassembly or move the latent stream to TCP/QUIC. Do
not send full-size raw latents as one UDP packet.

## Commands

Central side:

```bash
oasis-forge serve-central --latent-height 28 --latent-width 50 --fps 12
```

Laptop side, decoder:

```bash
cd edge/kvcraft-vae-metal-decoder
swift build -c release
.build/release/kvcraft-vae-metal --weights /path/to/kvcraft-vae-decoder-f16 \
  --udp-port 7777 --latent-height 28 --latent-width 50
```

Laptop side, bridge:

```bash
oasis-forge laptop-bridge --server-host <central-host> --player-id 0
```

Local action injection smoke test:

```bash
printf '{"buttons":1,"mouse_dx":0.5,"mouse_dy":0.0}' | nc -u -w0 127.0.0.1 7790
```

## Integration Point For The Real Generator

`run_central_server()` currently accepts a `LatentGenerator` protocol with two methods:

```python
push_action(action)
next_latent(player_id) -> bytes
```

The checked-in CLI uses `ZeroLatentGenerator` as a smoke test. The production server should replace
that with an adapter around the live JAX/KV Craft rollout state:

- maintain one authoritative action timeline per player
- advance the latent generator when enough actions are available
- return one raw f16 latent payload per player/chunk
- preserve sequence numbers and player routing

## Open Work

- Replace `ZeroLatentGenerator` with the real B300/H100 JAX latent generator.
- Wire real laptop input capture into UDP JSON updates on `127.0.0.1:7790`.
- Add packet loss handling, late action policy, and resync snapshots.
- Wire the Metal decoder output to a window/display path instead of timing-only logging.
- Add latent-stream fragmentation or switch to TCP/QUIC if full-size grids are required.
