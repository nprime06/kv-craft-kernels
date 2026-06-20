# 2026-06-20 - Split-serving protocol scaffold

Added the first concrete end-to-end seam for the intended deployment:

```text
B300/H100 central server: generate latents
Mac laptops: send actions upstream, decode latents locally
```

## Code

- `src/oasis_forge/streaming.py`
  - binary action packet format
  - binary latent packet format
  - central server relay loop
  - laptop bridge loop
  - zero-latent generator for smoke tests
- `oasis-forge serve-central`
  - listens for player actions
  - registers laptop endpoints
  - streams framed f16 latents back to each laptop
- `oasis-forge laptop-bridge`
  - accepts local JSON action updates
  - sends compact action packets to the central server
  - receives framed latents
  - forwards raw latent payloads to the local Swift/Metal decoder

## Important constraint

The existing Metal decoder expects one raw latent UDP datagram. Full default `45x80` latents are
`115200` bytes and do not fit safely in one UDP packet. The split-serving scaffold defaults to
`28x50` latents, which are `44800` bytes and fit while matching the measured `~12 FPS` local decode
target.

For full-size or larger latents, add fragmentation/reassembly or use a stream transport.

## Production integration point

The central server currently uses `ZeroLatentGenerator`. Replace it with an adapter around the live
JAX rollout:

```python
push_action(action)
next_latent(player_id) -> raw_f16_nhwtc_bytes
```

This keeps the networking and laptop decoder path independent from the details of the B300 model
runtime.

