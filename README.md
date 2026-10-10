# Strata on 4 GPUs (1 main + 3 computing tiers)

A fork of [eddoursul/Strata](https://github.com/eddoursul/Strata) (branch `custom`), which is itself a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata). The original project README now lives in [docs/README-upstream.md](docs/README-upstream.md); eddoursul's 3090+5070 Ti comparison is in [docs/COMPARISON.md](docs/COMPARISON.md). Everything here builds on eddoursul's expert-tier design; what we added is the part that scales it past two cards.

The short version: upstream Strata runs Qwen3.8-Flash-Next on one GPU and streams experts in from RAM. eddoursul's fork adds a second GPU that *computes* experts instead of just storing them, roughly doubling decode on a 3090 + 5070 Ti box. Both designs stop at exactly one extra GPU. Our rig is four RTX 5060 Ti 16 GB, so we converted the second-GPU machinery into an N-tier one and measured what each card buys. It's a lot, and none of it required touching the model.

## What we changed

Five things, in the order they landed.

1. **The tier array** (`src/core/expert_source.cpp`, `src/core/verify.cpp`). The per-layer plan, the grouped-expert graph, the prefetch stream and the completion flag were all built around one `SecondGpu`. Entries now bucket per owning tier, each tier submits its own captured graph, and the fan-in waits one 64-byte flag line per (token-group, tier) instead of one per group. The entry list the combine kernel reads stays merged, so the window graph keeps its shape.
2. **`--tier-gpus 1,2,3`** (`src/program/generate.cpp`). One init loop instead of a hard-coded pair. Each tier gets its own `ExpertCache`, its own `AdaptiveTier` (chained: every tier ranks after the caches above it and never duplicates them), its own prefetch deal, and its own health watch. The old `--second-gpu N` still works and means `--tier-gpus N`.
3. **N-way head split** (`src/core/split_head.*`). The original `--head-split` parked a share of the output-head rows on the second GPU only. `SplitHead::init` now takes an explicit `[start, end)` range and the window joins up to three parts plus the main card's, with the argmax merge picking across all of them.
4. **N-way prompt offload** (`src/prefill/prefill.cpp`). The batched prompt path used to hand all non-resident experts to one runner. Now a prompt expert goes to the runner whose cache holds it, experts no GPU holds are spread by hash, each runner stages from its own card's lent cache slots, and `gr_write` folds up to three fp16 partial sums into the residual in one pass. Prefill is where this shows.
5. **A routing-trace fix.** `--dump-routing` in the fork writes through a stdio FILE whose fd dies next to CUDA init: hundreds of pool calls, a cheerful `routing dumped (N records)` line (N is `drive.calls`, not records), and a 0-byte file. `make_profile.py` then "works" by quietly using only the base ranking. We replaced it with a counter-based dump from `ExpertDispatch::routed`, written with raw fd calls. Upstream main flushes its trace and doesn't have this shape, so this one is fork-specific.

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
- 256K context fits on the main card because the expert cache is `auto`: it measures free VRAM after the KV pool lands, so growing context just lets ranked experts fall into the tiers by itself. Config edit, not migration.

## Relationship to the parent repos

A PR to eddoursul/Strata `custom` is open from this repo's `custom-4gpu` branch and applies to `custom@3a199441`, which is still that branch's tip.

Upstream Niko1221/Strata deliberately gets no PR from us: it has no expert-tier code at all (its multi-GPU is a layer split) and its routing trace already flushes. Our work is a fork of a fork, and the right upstream is the middle one.

A 4×5060 Ti footnote for the vLLM crowd, since that's where this rig's day job runs: our production lane banks 59.7 t/s at 455K context with batch-2 concurrency against these numbers. Different trade, not a loss: this engine is one user, 256K, fast, and it never syncs attention across cards. Want concurrency and half-a-million tokens? That's tensor parallelism and collectives, and that's a different program.

## License

MIT, same as both parents. Model weights are not in this repo; each model's own license applies.
