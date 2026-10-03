# DESIGN-4GPU — 1 main + N tiers (P2), from code @3a19944

Working tree: local clone /home/john/.hermes/cache/scratch/strata-fork (branch play/4gpu);
rsync sources to ai-host ~/strata-4gpu/strata4 (separate from the untouched P1 tree);
build reuses the fetched llama.cpp via
  -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=~/strata-4gpu/strata/build/_deps/strata_llamacpp-src
P1 baseline to beat/parity-check: decode 49.1–60.8 t/s, pp 1982 @16K, ctx 131K,
main 6.74 GiB/2312 experts, tier 13.26 GiB/4540 experts.

## Protocol facts (verified in code, not docs)
- Window per layer has G token groups; per (l, grp): flag2 = h_flag2_ + grp*16 (64B lines),
  monotone want = l*G+grp+1. Verifier graph (verify.cpp ~787): fork g2s_ stream ->
  wait_flag_ge(m_flag2_+grp*16, want) -> fetch_listed_rows(list2_[grp], ...) -> join.
- Pool path (expert_source.cpp ~302-420): per routed entry kind = -1 CPU | 0 main-VRAM |
  1 PCIe | 2 tier. Tier groups: g2_slot/g2_start/g2_ent -> SecondGpu::submit (host-thread
  task GpuSubmitTask ~483). Tier rows_out kernel raises gpu2_flag = flag2[grp] when its
  last graph for the layer finishes; no-share path raises it host-side (n2g==0, and
  verify.cpp 1235 fallback for failed pools).
- Ownership: host_res2[layer*n_expert + e] = slot (>=0) or -1; prefetch slots coded
  -2 - p via gpu2->prefetched(layer, e). Slot pointers resolved INSIDE the tier device
  via cache_.device_slot(slot) at submit() plan-build time (p64 table).
- Tier cache is fed by: profile-ranked admit at load (generate.cpp 2336) + AdaptiveTier
  swaps (tier2.init at 2349, paced on gpu2.done_event(), own thread at 2356).
- Prefetch: candidate set from verifier's next-layer prediction (predict_/set_predict),
  experts NO GPU holds; copied on the tier's pre_s_ stream during main's ~0.28 ms serial
  gap; computed as plan part 1 after their copies land.
- Prompt path: prefill::ExpertRunner offload (generate.cpp 2413) computed on gpu2 with
  &gpu2.cache(); prefill loan slots pbuf.loan2 (cache ptr).
- Head: split_head.cpp takes [--head-split F] share of the output head's rows (0..F of
  rows on the tier; P1 ran 0).
- SecondGpu internals to replicate per tier: s_/ev_, h_x_/m_x_ + h_plan_/m_plan_ (pinned
  portable mapped), d_xq_/d_plan_/d_scratch_/d_rows_/d_count_, cache_, graphs_,
  pre_s_/pre_ev_/d_pre_/pre_ids_/pre_layer_, DeviceScope(dev, main) per call.

## Conversion (per file)
1. expert_source.hpp: `SecondGpu* gpu2` -> `std::vector<TierView> tiers`,
   TierView { SecondGpu* gpu; const int32_t* host_res; cudaEvent_t done; }. kind[] becomes
   -3-t for tiers (t>=1: -3 => tier0 share… keep -1 CPU, 0 vram, 1 pcie, and kind 2+t for
   tier t). Per-tier plan arrays (slot/start/ent) sized 128 each (n<=128 entries total).
2. expert_source.cpp ~302: slot2(e) -> tier_of(e) loop over host_res arrays (O(T));
   gpu2_used decision becomes per-tier share decision (min-bytes test against THIS tier's
   bytes). One submit task per tier per group (GpuSubmitTask takes tier ptr).
3. verify.cpp: per (grp, tier) flag slot: h_flag2_ grows to T lines per grp
   (stride grp*16*T + t*16); wait graph: loop T sequential wait_flag_ge + per-tier
   list2_ slices; fetch_listed_rows per list. Empty-share raise: host-side per tier.
4. generate.cpp: --tier-gpus "1,2,3" (alias --second-gpu N => 1 tier); init loop:
   SecondGpu::init(dev_i, main_dev...) + init_prefetch + cache open sized by
   --tier-gib list (default all-2GiB each); profile admit: rank windows DISJOINT per tier
   (skip pairs owned by main or earlier tiers); N AdaptiveTiers each owning one cache,
   paced on its own done_event; N ExpertRunners for the prompt path; pbuf loans per tier;
   --head-split spread F across tiers proportionally.
5. split_head.cpp: share vector per tier.
6. prefill/experts.cpp: unchanged per-runner (already device-scoped) — just instantiate N.
7. serve/server.py + config: pass --tier-gpus through; our-ud-q4k.json gets
   --tier-gpus 1,2,3 + per-tier gib + head-split 0.3 spread.

## Prefetch assignment (the one real design choice)
Ranked no-GPU-has candidates are dealt round-robin to tiers (tier t takes rank i where
i mod T == t), capped by each tier's prefetch slots/bytes. Rationale: candidates share
one ranking from the router prediction; slicing preserves likelihood order per leg and
spreads copy pressure across legs (x8/x4/x4/x4 gen5). Alternative (all prefetch to
tier0) rejected: serializes one leg during the other two's compute windows.

## Gates per increment (never skip)
- Inc1 (TierSet wrapper, still 1 tier): boot parity — same flags, same requests, decode
  within run-to-run noise of P1, needle spot-check answers identical.
- Inc2 (2 tiers): greedy canonical check via tools/canonical_xcheck.py where applicable;
  argmax agreement on 3 fixed prompts vs P1 text (tiers change placement, so expect
  >= llama.cpp CPU-vs-CUDA class agreement, not bit-exact).
- Inc3 (3 tiers + head-split + 200K ctx): bench table (short decode / 23K decode / 16K
  pp / deep 128K decode), then compare vs lane bank.

## Risk notes
- cap_ per tier = max_window*k; three submits per group per layer = 3x host-plan writes;
  pinned staging per tier — RAM cost ~3x(16KB plan + window xq) — negligible.
- The 'submit spins cudaEventQuery with _mm_pause' pattern: with 3 tiers the host pool
  thread serializes submits — each waits ITS OWN last share only; keep submit calls on
  the existing pool task order to not add latency.
- Flag2 memory-order: existing seq_cst fences around volatile writes — reuse verbatim
  per tier; do not coalesce writers into one counter (device-side sys-scope atomics to
  mapped host from 3 devices is unproven on this driver; separate lines are proven).
- NEVER touch: ~/bucko-exp-20260925, :8000/:30001 containers, los units, push to Bucko.
