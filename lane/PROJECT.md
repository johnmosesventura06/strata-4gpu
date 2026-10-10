# Strata-4GPU — play project (started 2026-10-04)

Goal: convert the eddoursul/Strata `custom` fork's "second GPU as expert tier" design
into a **1 main + 3 tier** engine on the 4× RTX 5060 Ti rig (ai-host). Play/test only —
the golden lane (:8000) and the uncensored arm are NOT touched; lane relaunch stays
owner-gated. If a full-4-card run needs the lane down, that's John's call.

## What the fork does (code-verified, from ~/strata-4gpu/strata @3a19944)
- Main GPU runs 100% of attention/trunk/MTP serially + its own hot expert cache.
  Header measured the main card's serial part at ~0.28 ms/layer.
- Second GPU (`src/core/second_gpu.cpp`, `include/strata/core/second_gpu.hpp`):
  own ExpertCache + prefetch slots; per layer a captured CUDA graph runs
  `native_expert_grouped` (same kernels as main's hit path) on ITS experts, writes
  output rows into mapped host memory (16-byte stores) and raises a flag main waits on.
  Prefetch copies the next layer's likeliest experts in the RAM-idle gap (~12 MiB/layer).
- CPU pool computes the misses (experts neither GPU holds).
- `split_head.cpp`: fraction (--head-split, default 0.45) of output-head rows on the
  second GPU. `--second-gpu-min-mb 4`: tier only takes a layer's share above that.
- Prompt path: prefill/experts.cpp runs prompt-path experts the main cache lacks on
  the tier too (--no-prompt-offload disables).
- Configs (examples/*.json): IQ3_S / UD-Q4_K_XL, 200192 ctx, kv int8, spec 4 (MTP),
  spec-lookup 16, adapt-every 1, prefill 16384.

## Conversion surface (what must change for 3 tiers)
1. `SecondGpu` → per-device array: plan staging (h_plan_/d_plan_), graph set, ev_,
   prefetch stream/slots are all single-device today.
2. Layer-entry split across tiers by expert ownership (AdaptiveTier ranking 2-card → N-card).
3. Completion: one host flag → shared atomic counter (d_count_ pattern extends).
4. Prefetch pacing per PCIe leg (our links: x8/x4/x8/x4 gen5 CPU-direct; their 3090
   already ran x4 gen4, so per-leg pacing is proven, ours is 2× the BW).
5. head-split → N-way.
6. KV + trunk stay on main — context ceiling is main-card VRAM (16 GB → est. 128–160K
   with int8 KV; q4_0/k8v4 KV options exist in the fork for stretching).

## Phases
- P0 DONE (10-04 ~05:20): weights+pack+MTP+engine build; P1 baseline decode 49-61 / pp 1982.
- P1 DONE: see vault strata-4gpu.md §P1 baseline.
- P2 DONE (10-04 06:2x): multi-tier conversion shipped — --tier-gpus "1,2,3", per-tier
  plan/submit/flags (merged entry list, fan-out only flags), round-robin prefetch deal,
  disjoint rank windows per tier, AdaptiveTier chaining, drive_watch per tier.
  LADDER: 1 tier 49-53 -> 2 tiers 72-78 -> 3 tiers 80-89 t/s decode; prefill ~1980 flat.
  Correctness: needle@20K, code gen, Inc1 parity gate all passed. play/4gpu commits
  bf192b3/cc3497b/7c783b7; mirror ~/strata-4gpu/strata4 on ai-host; 3-tier live on :8093.
- P3 NEXT: deep-ctx sustained decode (100K+), q4_0/k8v4 KV for ctx > 131K, N-way head-split
  to free main VRAM, retune min-mb now that CPU pool is near-idle; compare vs lane bank.

## Facts / constraints
- Rig: 9950X, 128 GB DDR5-5000 (~62% peak bw clean), driver 615.71.09-p2p, root 990 Pro.
- g++ 15.2 — if nvcc 13.4 rejects it as host compiler, apt gcc-14/g++-14 +
  -DCMAKE_CUDA_HOST_COMPILER.
- UD-Q4_K_XL needs ~81 GB host RAM for experts (fork's measured) + 29 GB PLE table
  (inside shard 2/3, mmap'd from NVMe) — fits 128 GB.
- Unsloth repo layout: shards under UD-Q4_K_XL/ subdir (README's flat naming was wrong).
- Download: ~/strata-4gpu/bin/dl_udq4kxl.py (token read internally from ~/.bashrc,
  never echoed); resume-safe via .cache incomplete files.
- NEVER: touch qwen38 lane containers, los units, or push anything to origin Bucko.

## Layout
- repo:      ~/strata-4gpu/strata        (custom branch @3a19944)
- weights:   ~/models/UD-Q4_K_XL         (990 Pro)
- data dir:  ~/strata-4gpu/Strata-data   (packs/mtp per examples/README)
- logs:      ~/strata-4gpu-download.log, ~/cuda-install.log
- build:     ~/strata-4gpu/strata/build (cmake -G Ninja, CUDA arch 120)
