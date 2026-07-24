# SM110 Split-KV Decode Attention — Quick Reference

Hand-written single-query GQA decode attention for Cosmos3 Reasoner on Thor.
Partitions the KV sequence into a fixed `SPLITS=24` grid so long caches still
occupy Thor's modest SM count. Partials carry online-softmax `(m, l)` state
and are merged in a second kernel.

Reference code:
[`reference-kernels/nvidia/blackwell-thor/cuda/cosmos3_reasoner_attn/`](../../../../../reference-kernels/nvidia/blackwell-thor/cuda/cosmos3_reasoner_attn/).

Upstream pin: FlashRT
`e50bec2867b0bdac1d1f4cdcda1a6e4198a8743d`.

## When to apply this recipe

- Workload: decode, single query, GQA; KV length long enough that
  `num_heads` alone underfills the device
- Target: Jetson Thor / SM110 (and similarly low-SM embedded GPUs)
- Integration: CUDA Graph friendly — read sequence length from a device
  `int32`, not a host capture-time constant

## Recipe (in this order)

### 1. Split along KV, not only along heads

```cpp
constexpr int SPLITS = 24;  // fills Thor SMs on long KV
dim3 grid(num_heads, SPLITS);
```

One block owns `(query_head, split)`. Chunk bounds:

```text
chunk   = ceil(len / SPLITS)
s_begin = split * chunk
s_end   = min(len, s_begin + chunk)
```

### 2. Emit unnormalized partials + softmax state

Each split block runs FP32 online softmax over its KV chunk and writes:

- `part_acc[head, split, :]` — unnormalized output accumulator
- `part_ml[head, split, 0/1]` — `(m, l)` state

### 3. Merge splits with a second kernel

Reconstruct the global `(m, l)`, rescale each partial, and write the final
BF16 output. Treat merge as part of the recipe — split without merge is not a
complete attention.

### 4. Keep length device-side for CUDA Graph

Pass `const int* len_ptr` so graph replay tracks a growing KV prefix without
rebuilding a fixed-window mask each step. The same pattern applies to the
fused RoPE + KV-append helpers in the same source file.

## What not to do

- Do not copy a data-center FMHA launch grid that assumes hundreds of SMs and
  enough head parallelism alone.
- Do not bake host-side `len` into a captured graph if the cache grows across
  steps.
- Do not reuse B200/SM120 attention timing as Thor evidence.

## Tags

```text
architecture: sm110
product: jetson-thor
operator: flash-attention
workload: decode, single-query, gqa
dtype: bf16
optimization: split-kv, online-softmax, cuda-graph-safe
status: historical-snapshot
source: FlashRT
```
