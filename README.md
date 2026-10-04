# Strata on 4 GPUs (1 main + 3 computing tiers)

A fork of [eddoursul/Strata](https://github.com/eddoursul/Strata) (branch `custom`), which is itself a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata). The original project README now lives in [docs/README-upstream.md](docs/README-upstream.md); eddoursul's 3090+5070 Ti comparison is in [docs/COMPARISON.md](docs/COMPARISON.md). Everything here builds on eddoursul's expert-tier design; what we added is the part that scales it past two cards.

The short version: upstream Strata runs Qwen3.8-Flash-Next on one GPU and streams experts in from RAM. eddoursul's fork adds a second GPU that *computes* experts instead of just storing them, roughly doubling decode on a 3090 + 5070 Ti box. Both designs stop at exactly one extra GPU. Our rig is four RTX 5060 Ti 16 GB, so we converted the second-GPU machinery into an N-tier one and measured what each card buys. It's a lot, and none of it required touching the model.

## What we changed

Six things, in the order they landed.

1. **YaRN / rope-scaling, ported from upstream main** (`src/program/generate.cpp` and every rope site). The fork merged 0.1.30's kernels but only the non-scaled CLI surface: every rotation computed `theta = pos * pow(base, -pair)` from a bare float. Upstream main resolves a process-wide `RopeScaling` (none|linear|yarn) once at startup — K in the cache is post-RoPE, so one run uses one scaling — and threads it through the decode tables and every analytic kernel. We ported that whole plumbing (prefill `rope()`, `native_rope`, `native_qsa` norm + indexer, the decode table builder), added the CLI (`--rope-scaling`, `--rope-scale`, `--yarn-orig-ctx`, ...), and swapped `qsa_select.cu` to upstream's version because the fork's register top-k was capped at exactly the trained 262,144 cells — the new dispatcher falls back to its histogram kernel on GPUs without thread-block clusters (all consumer Blackwell), which is what makes 524K decode windows work here. See "Past the trained context" below.
2. **The tier array** (`src/core/expert_source.cpp`, `src/core/verify.cpp`). The per-layer plan, the grouped-expert graph, the prefetch stream and the completion flag were all built around one `SecondGpu`. Entries now bucket per owning tier, each tier submits its own captured graph, and the fan-in waits one 64-byte flag line per (token-group, tier) instead of one per group. The entry list the combine kernel reads stays merged, so the window graph keeps its shape.
3. **`--tier-gpus 1,2,3`** (`src/program/generate.cpp`). One init loop instead of a hard-coded pair. Each tier gets its own `ExpertCache`, its own `AdaptiveTier` (chained: every tier ranks after the caches above it and never duplicates them), its own prefetch deal, and its own health watch. The old `--second-gpu N` still works and means `--tier-gpus N`.
4. **N-way head split** (`src/core/split_head.*`). The original `--head-split` parked a share of the output-head rows on the second GPU only. `SplitHead::init` now takes an explicit `[start, end)` range and the window joins up to three parts plus the main card's, with the argmax merge picking across all of them.
5. **N-way prompt offload** (`src/prefill/prefill.cpp`). The batched prompt path used to hand all non-resident experts to one runner. Now a prompt expert goes to the runner whose cache holds it, experts no GPU holds are spread by hash, each runner stages from its own card's lent cache slots, and `gr_write` folds up to three fp16 partial sums into the residual in one pass. Prefill is where this shows.
6. **A routing-trace fix.** `--dump-routing` in the fork writes through a stdio FILE whose fd dies next to CUDA init: hundreds of pool calls, a cheerful `routing dumped (N records)` line (N is `drive.calls`, not records), and a 0-byte file. `make_profile.py` then "works" by quietly using only the base ranking. We replaced it with a counter-based dump from `ExpertDispatch::routed`, written with raw fd calls. Upstream main flushes its trace and doesn't have this shape, so this one is fork-specific.

## The speed ladder

RTX 5060 Ti ×4 (PCIe gen5, x8/x4/x8/x4 CPU-direct), 9950X, 128 GB DDR5-5000, 131K int8 KV unless noted, `UD-Q4_K_XL`, MTP + prompt-lookup on, single stream, greedy, engine-measured tokens/s.

| Step | Decode (sustained) | Notes |
|---|---|---|
| main + 1 tier | 49–53 t/s | matches the fork's own issue-#392 dual-5060-Ti report (50–60 on a much weaker box) |
| main + 2 tiers | 72–78 t/s | tiers fill from the profile ranking, disjoint windows |
| main + 3 tiers | 80–89 t/s | ~65% of all 24,576 experts live in VRAM; the CPU pool nearly starves |
| + bound draft head | 91–94 t/s | `rt/draft_vocab.bin` was missing on our build; see pitfalls. Bigger win than the head split |
| + head-split 0.40–0.75 | 94–99 t/s | one card holding the share measured best; 3-way spread landed equal, kept as the right shape |
| + N-way prompt offload | 92–94 sustained | 99–110 on short turns, best burst 110.8 |

Prefill, cold: 16K prompts went 1,878 → 2,159 t/s with the N-way prompt path. A 220K-token prompt reads at 2,420 t/s (was 2,129–2,175) and a needle buried at 211K still comes back verbatim. Warm-prefix reuse is another tier of its own: the fork's stashed conversation cache replayed a 94K and a 211K prompt in about 12 s each.

One honest footnote. Between-boot numbers wobble about ±3%, and generated-text tokens/s wobble more, because prompt-lookup accept rate depends on what the model wrote. Same-boot A/B pairs are the real signal; a lone 90s or 100s reading is ballpark. The 3-way head in particular measured +2.3 over its own control, inside noise; it stays because it's the architecture, not because this box felt it.

## Past the trained context: 524K with YaRN

The model was trained to 262,144. With `--rope-scaling yarn --rope-scale 2.0 --max-context 524288`
this lane boots and serves at twice that, on the same 4 cards, same `UD-Q4_K_XL` pack, int8 KV:

| Measurement @ 524K | Number |
|---|---|
| cold prefill, 16K prompt | 1,720 t/s first touch, 1,886 steady |
| cold prefill, 281K prompt | 1,418 t/s |
| decode at depth (225K–312K conversation) | 84–111 t/s |
| warm replay of a 312K prompt (checkpoint cache) | 12 ms (312,134/312,135 tokens served from cache) |
| identity gate: `--rope-scale 1.0` @196K | 98.2 t/s decode — bit-band of unscaled, the port is neutral when off |

Two budget facts the 524K boot teaches. First, the int8 KV pool (~3.7 GB) lands on the main card
before the auto expert cache sizes itself, so ranked experts drain to the tiers: main holds ~202
slots. Measured cost of that squeeze is ~3% on prefill and ~5% per halving on decode — the tiers
catch everything, the bytes don't. Second, the real 524K cost is the prompt chunk: the staging
buffers scale with `--prefill`, and with a starved main cache a 24,576 chunk cannot allocate.
Clamped to 4,096, 16K prefill reads 2,159 → 1,778 t/s *with the expert cache untouched* (A/B on
the 256K boot) — the four-sub-chunk loop overhead, not residency, is the 18%. Raising the chunk
back means freeing KV-side bytes (k8v4 is the lever; we have not gaunted it).

## Running it

Build deps: cmake, ninja, a CUDA 13 toolkit (we used 13.3), and an NVIDIA driver new enough for it. From the repo root:

```
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target strata
```

Weights: `unsloth/Qwen3.8-Flash-Next-GGUF`, the `UD-Q4_K_XL/` four shards, into `Strata-data/models/UD-Q4_K_XL/` (the fork's examples README says flat filenames; they actually live under that subdirectory). Pack:

```
python3 tools/iq_pack.py --out ../Strata-data/packs/ud-q4_k_xl \
  --gguf ../Strata-data/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
```

MTP draft layer: `tools/mtp_fetch.py fetch` pulls ~5 GB of `mtp.*` tensor ranges from the BF16 checkpoint over HTTP without downloading the checkpoint, then `tools/mtp_pack.py` and `tools/mtp_rt.py` build the runtime folder. Then the step nobody documents: copy `data/draft_vocab.bin` into that runtime folder. Without it `--head-split` silently no-ops and you lose about 10% decode for a reason you'll never guess from the logs.

Serve with `examples/ud-q4_k_xl-4gpu.json` (or your own) on whatever port is free:

```
python3 serve/server.py --engine strata --config examples/ud-q4_k_xl-4gpu.json --port 8090
```

## Pitfalls we hit so you don't

- `--vram-reserve-mib` defaults to 700. Big contexts need ~1500 or the prefill cuBLAS side-stream handles can't allocate and the engine dies before READY, printing what looks like a progress line (`prefill gemm: side streams` is an error string).
- After killing a loaded engine, its ~71 GiB pinned arena takes 10–40 s to release. Relaunching immediately fails with `cudaMalloc(1.48 GB) for the weight arena failed`. Wait for the cards to read idle.
- `--head-split 1.0` is silently refused (the guard wants `< 1.0`), and the refusal is one buried line at boot.
- The `set_gpu2`-era naming is everywhere; "second GPU" now means "a tier". We left the names alone, a rename is a PR of its own.
- At 524K, `--prefill` must drop to 4,096 and `--vram-reserve-mib` to stay around 1,500: raising the reserve to buy bigger staging instead starves the 1.4 GB weight arena and the boot dies before READY.
- Consumer Blackwell (sm_120) has **no thread-block clusters**. Upstream's 512K selection story leans on a cluster kernel; on these cards the dispatcher falls back to the histogram kernel — correct, just slower per call. Nothing to configure, but don't chase "why isn't the cluster path used" — it can't be.
- 256K context fits on the main card because the expert cache is `auto`: it measures free VRAM after the KV pool lands, so growing context just lets ranked experts fall into the tiers by itself. Config edit, not migration.

## Relationship to the parent repos

Two PRs to eddoursul/Strata `custom` are open from this repo: `custom-4gpu` (the N-tier work) and `yarn-500k` (the rope-scaling port on top of it). Both apply to `custom@3a199441`, which is still that branch's tip.

Upstream Niko1221/Strata deliberately gets no code PR from us: it has no expert-tier code at all (its multi-GPU is a layer split), and the rope machinery we ported is their own main-branch code. What they don't have is field data for consumer Blackwell at 524K — the cluster-less fallback path, measured — which we filed as an issue. Our work is a fork of a fork, and the right upstream is the middle one.

A 4×5060 Ti footnote for the vLLM crowd, since that's where this rig's day job runs: our production lane banks 59.7 t/s at 455K context with batch-2 concurrency against these numbers. Different trade, not a loss: this engine is one user, 256K, fast, and it never syncs attention across cards. Want concurrency and half-a-million tokens? That's tensor parallelism and collectives, and that's a different program.

## License

MIT, same as both parents. Model weights are not in this repo; each model's own license applies.
