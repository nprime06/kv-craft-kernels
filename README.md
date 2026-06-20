# WM Serving Kernels — KV Craft

GPU kernel optimization for **real-time multiplayer world-model serving**. Current target:
**KV Craft** (open JAX multiplayer Minecraft WM — (base WM repo, internal)) — make its
per-step inference stream fast on **B300 (sm_100)** and **H100 (sm_90)**.

## Scope (collaboration)

- **This repo / us: the KERNELS.** Profile the model's hot kernels (multiplayer joint attention,
  per-player FFN/attn, VAE decode, AdaLN, RoPE) and optimize them for fast streaming on B300/H100.
  JAX-native: XLA cuDNN-FMHA, Pallas-GPU kernels, JAX FFI to FA3/CUTLASS.
- **Collaborators: the interactive harness + network/systems design** (real-time playable loop,
  multiplayer netcode, resync, cross-player KV sharing).

## Status (2026-06-20)

Harness + profiler workflow built and the cross-SM port mechanics proven on PyTorch synthetic
kernels (a methodology demo). **Pivoted to KV Craft (JAX);** bringing it up on the B300 to profile
real shapes, then kernel opt moves to JAX-native. Live state: `knowledge/loop_state.md`. Findings:
`knowledge/episodes/`.

Client-side edge work is now tracked separately under `edge/`: `edge/kvcraft-vae-metal-decoder/`
is a native macOS Swift/Metal KV Craft VAE decoder for the split-serving path where the server
streams latents and the Mac decodes RGB locally. See
`edge/kvcraft-vae-metal-decoder/Docs/EDGE_INFERENCE_OPTIMIZATIONS.md`.

Split-serving glue is now in `src/oasis_forge/streaming.py`: B300/H100 servers receive laptop
actions, generate latent chunks, and stream framed f16 latents back to each laptop; each laptop
bridge strips the frame header and forwards raw latent bytes to the local Metal decoder. See
`knowledge/semantic/split_serving_architecture.md`.

---

### Legacy framing (being re-targeted to KV Craft)

Originally an autonomous **port-and-optimize** loop for B300 (sm_100) → H100/H200 (sm_90),
modeled on `amd-kernel-forge`. The harness, two-tier gate, ledger, roofline calc, and profiler
(`prof.py`) all carry over; the kernel *implementation* moves from Triton to JAX/Pallas.

## Why this exists

End state: a multiplayer diffusion world model where **each user gets their own H100
cluster**. The model is served on B300 first; the kernels must then run on Hopper. A
Blackwell kernel using sm_100 features won't necessarily be optimal — or even compile —
on sm_90, so we need a loop that re-derives each hot kernel for Hopper and proves it
didn't break the model.

## The two models

| Role | Model | Precision | Hardware | Status |
|------|-------|-----------|----------|--------|
| **Prototype** | Oasis-500M (`camenduru/oasis-500m`, ST-DiT + ViT-VAE) | FP16 | single H100 / A100 / 4090 | open, runnable |
| **Ship** | internal 14B Wan2.1 DMD driving world model | FP16 → (opt) FP8 | B300 source → H100 target | not yet served |

The 500M prototype is FP16 (matches the real source precision) and rolls out autoregressively
(compounding-drift). **But it diverges from the ship model more than the shared name suggests**
(verified by reading both — see `knowledge/semantic/model_facts.md`):

| Kernel | 500M prototype | 14B ship | Transfers? |
|--------|----------------|----------|------------|
| Attention | axial, SDPA, head_dim 64 | full-3D, head_dim 128 | **NO (different in kind)** |
| AdaLN | SiLU+Linear→6·h, additive cond | same op, wider | **YES — best target** |
| VAE decode | ViT transformer (SDPA) | Wan-VAE 3D causal conv | **NO (different in kind)** |
| Sampler | DDIM 10-step | DMD 4-step iterative | harness only |

So Oasis-500M faithfully proves the **harness, gate, ledger, loop, and the AdaLN port** — but it
is **not** a kernel-transfer proxy for attention or VAE. See "Prototype choice" below.

### Prototype choice (research finding)

For a kernel-port harness whose ship target is a 14B **Wan2.1** DMD world model, the better open
prototype is **Wan2.1 itself** — specifically the distilled streaming variants **CausVid /
Self-Forcing on Wan2.1-1.3B** (4-step, block-causal, KV-cached, ~17 FPS on a single H100). Those
match the ship model's *architecture* (full-3D head_dim 128, Wan-VAE conv3d) **and** its few-step
autoregressive regime — which Oasis-500M (axial hd64, ViT-VAE, DDIM-10) does not. Recommendation:
keep Oasis-500M to bring up the harness cheaply, then point harvest at Wan2.1-1.3B/Self-Forcing
for the attention + VAE kernels that actually ship. Open question for you — see chat.

## The three hot kernels (harvest targets)

1. **DiT attention** — the O(n²) prime target. **77-85% of each DiT forward pass** at the ship
   model's token counts (~33k @480p, ~76k @720p). 14B = full-3D head_dim 128 (FA3's sweet spot),
   block-causal + KV cache; 500M = axial SDPA head_dim 64.
2. **AdaLN modulation** — action-conditioned shift/scale/gate per block. HBM-bandwidth-bound;
   Triton fusion is a validated 3.2-3.4×. Cheapest win and the best prototype→ship transfer.
3. **VAE decode** — latent → RGB. **Co-priority with attention**, not an afterthought: because the
   distilled model runs the DiT only ~4×, decode (step-count-independent) rises to **30-40% of
   end-to-end latency** in the few-step regime. 14B = Wan-VAE 3D causal conv (compute-bound, cuDNN
   channels_last_3d); 500M = ViT transformer VAE.

Final list + exact shapes come from a **profile on B300/H100**, not this README. Two cross-cutting
facts from the research: keep **attention in FP16/BF16** (FP8-attention drift compounds across the
autoregressive rollout — RMSE ~48× worse even mitigated); spend the **FP8 aggressive tier on the
dense FFN/QKV GEMMs** instead. Amortize KV across **chunks**, not the 4 denoising steps.

## The loop (3 stages)

```
1. HARVEST   instrument the rollout, dump {inputs, golden_output, shape, dtype}
             at each hot call site, at representative DMD steps.
             golden = high-precision (BF16/FP32) reference, NOT the FP16 path.
                 |
2. PORT+OPT  per kernel, agent proposes an sm_90 kernel:
               - STRUCTURAL tier (FP16->FP16): fusion + TMA + wgmma, ~zero precision
                 loss. Correctness is BINARY (allclose at reassociation tolerance).
               - AGGRESSIVE tier (FP16->FP8): opt-in, bigger speedup, must clear the
                 trajectory gate. Reach for it only where the drift budget allows.
             compile for sm_90 -> tier-1 correctness -> benchmark on H100 -> keep if
             faster AND correct.
                 |
3. GATE      swap winners into the stack -> full autoregressive rollout on H100 ->
             two-tier gate:
               tier 1 (per-kernel): allclose vs golden  [cheap, inner loop]
               tier 2 (trajectory): N-frame rollout, latent MSE + perceptual delta
                                    vs reference  [expensive, gates integration]
```

### The gate insight (why this is tractable)

For an autoregressive world model, per-kernel error **compounds across frames** — a kernel
can pass an isolated allclose and still turn the scene to mush by frame 80. So tier 2 is
mandatory, not optional. Two precision tiers give two clean rules:

- **Structural FP16→FP16:** no precision loss, so any tier-1 failure is a *real bug*, not
  a tolerance judgment. Correctness is binary and cheap.
- **Aggressive FP16→FP8:** scored against the inequality `drift(fp8_port) <= drift_budget`.
  Because the shipped B300 deployment is itself an approximation, the budget can be pinned
  to "no worse than what we already ship."

This is `feedback_optimize_the_scored_objective` applied to a recurrent system: the scored
objective is **trajectory quality**, never single-call error.

## Layout

```
agents/agent_prompt.md      # port+optimize methodology (the agent's playbook)
problems/<kernel>/
  problem.yaml              # kernel id, source arch, target arch, tier, shapes (from harvest)
  task_files/               # reference.py, eval.py, submission.py  (populated by harvest)
harvest/
  hooks.py                  # call-site instrumentation -> golden dumps  (STUB until rollout exists)
  README.md
edge/
  kvcraft-vae-metal-decoder/ # macOS Swift/Metal client-side KV Craft VAE decoder
src/oasis_forge/
  config.py                # sm_90 Hopper hardware config + port config
  streaming.py             # central-server/laptop split-serving protocol + bridge loops
  ledger.py                # flat JSONL attempt ledger (no DB)
  gate.py                  # two-tier drift gate
  remote.py                # H100 SSH executor  (STUB until box exists)
  cli.py                   # harvest / solve / gate / bench
knowledge/
  semantic/                # hardware_sm90.md, port_strategy.md, research_sources.md
  episodes/                # campaign logs (agent-written)
  solutions/               # best kernel per problem (auto-saved)
scripts/
  loop.sh                  # continuous loop runner
  monitor.sh               # live dashboard
```

## Status

**Design + skeleton.** Harvest hooks and the H100 remote executor are stubs — they wire up
the moment (a) Oasis-500M does one rollout on a box and (b) there's an H100 to benchmark on.
Everything precision/gate/ledger-side is real and CPU-testable.

## Split-Serving Smoke Test

Central server, on the B300/H100 side:

```bash
oasis-forge serve-central --latent-height 28 --latent-width 50 --fps 12
```

Laptop:

```bash
cd edge/kvcraft-vae-metal-decoder
swift build -c release
.build/release/kvcraft-vae-metal --weights /path/to/kvcraft-vae-decoder-f16 \
  --udp-port 7777 --latent-height 28 --latent-width 50
```

In a second laptop terminal:

```bash
oasis-forge laptop-bridge --server-host <central-host> --player-id 0
```

Local input capture should send JSON action updates to the bridge:

```bash
printf '{"buttons":1,"mouse_dx":0.5,"mouse_dy":0.0}' | nc -u -w0 127.0.0.1 7790
```

The checked-in central loop currently emits zero latents as a protocol smoke test. Replace
`ZeroLatentGenerator` with the live JAX/KV Craft latent generator to drive real frames.

## Open dependencies (gate the build)

- [ ] A box to run Oasis-500M (one H100/A100/4090) — unblocks harvest.
- [ ] B300 serving of the 14B model — unblocks real golden tensors + FP8 numerics.
- [ ] An H100/H200 to benchmark the port target on.
