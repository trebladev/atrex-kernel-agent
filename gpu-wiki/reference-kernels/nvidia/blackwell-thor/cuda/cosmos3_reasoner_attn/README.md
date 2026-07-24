# Cosmos3 Split-KV Decode Attention

> **Usability status:** `historical-snapshot`
>
> Extracted from FlashRT. Compile requires `-DFLASHRT_HAVE_COSMOS3_REASONER`.

Single-query GQA decode attention with fixed `SPLITS=24` KV partitioning to
fill Thor SMs on long caches. Length is read from a device `int32` so a CUDA
Graph can replay over a growing KV prefix. The same file also contains fused
RoPE + KV-cache append helpers (BF16 and FP8-KV variants).

---

| File | Description |
|------|-------------|
| [cosmos3_reasoner_attn.cu](cosmos3_reasoner_attn.cu) | Split kernel, merge kernel, RoPE/KV append |
| [cosmos3_reasoner_attn.cuh](cosmos3_reasoner_attn.cuh) | Launch APIs |
| [NOTICE](NOTICE) | Upstream commit and Apache-2.0 provenance |

## Hardware and scope

- Hardware: Jetson Thor / SM110 (low SM count relative to data-center GPUs).
- Workload: decode, single query, GQA (`NH=16`, `NKV=8`, `HD=128` in comments).
- Pattern: one block per `(query_head, KV_split)` → online-softmax partials →
  merge.

## Related docs

- Optimization card:
  [sm110-split-kv-decode-attention.md](../../../../../docs/nvidia/blackwell-thor/kernel-opt/cuda/sm110-split-kv-decode-attention.md)
