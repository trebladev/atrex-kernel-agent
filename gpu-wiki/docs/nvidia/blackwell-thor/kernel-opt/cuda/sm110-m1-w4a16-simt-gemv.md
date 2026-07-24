# SM110 M=1 W4A16 SIMT GEMV — Quick Reference

Decode-stage `M=1` linear as a SIMT GEMV instead of an NVFP4 Tensor Core GEMM.
One warp owns one output row; packed E2M1 weights stream from DRAM; the BF16
activation vector stays in L1/L2 without shared-memory staging.

Reference code:
[`reference-kernels/nvidia/blackwell-thor/cuda/cosmos3_reasoner_gemv/`](../../../../../reference-kernels/nvidia/blackwell-thor/cuda/cosmos3_reasoner_gemv/).

Upstream pin: FlashRT
`e50bec2867b0bdac1d1f4cdcda1a6e4198a8743d`.

## When to apply this recipe

- Workload: decode `M=1` W4A16 (or similar low-bit weight) linear layers
- Target: Jetson Thor / SM110 with Blackwell FP4 conversion
  (`__CUDA_ARCH__ >= 1000`)
- Decision boundary: when Tensor Core tile overhead exceeds the gain because
  the bottleneck is weight bandwidth, not MMA throughput

## Recipe (in this order)

### 1. Prefer SIMT GEMV over Tensor Core when M=1

At `M=1` the math is a dot product per output row. A warp-specialized TC GEMM
pays layout, staging, and epilogue costs that do not buy occupancy on a
single-row problem. Stream weights and FMA in SIMT instead.

### 2. Pack E2M1 two-per-byte with block scales

```text
weights: [N, K/2] uint8   (low nibble = even k)
scales:  [N, K/16] BF16   (one scale per 16 weights)
activation: [K] BF16
K % 16 == 0
```

### 3. Use hardware FP4→half conversion on Blackwell

```cpp
#if __CUDA_ARCH__ >= 1000
  __nv_fp4x2_storage_t code = ...;
  __half2_raw wr = __nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1);
#else
  // LUT fallback for older arch builds
#endif
```

Keep a LUT path only for non-Blackwell compile targets; production Thor
builds should hit the hardware convert.

### 4. Skip SMEM staging for the activation vector

The activation is a few KB shared by every block. Reading it from global
memory lets L1/L2 supply it; copying into SMEM adds traffic and a barrier
without increasing weight bandwidth.

Launch shape: `blockDim = (32, rows_per_block)` with one warp (`threadIdx.x`)
per output row (`threadIdx.y`).

## What not to do

- Do not force a large TC tile “because Blackwell has Tensor Cores” on pure
  `M=1` decode.
- Do not introduce SMEM staging for a tiny, highly reused activation vector
  unless profiling shows L1 thrashing.
- Do not confuse this W4A16 SIMT path with block-scaled NVFP4 Tensor Core
  GEMM tactics used for larger M.

## Tags

```text
architecture: sm110
product: jetson-thor
operator: gemv
workload: decode, m1
dtype: nvfp4-weight, bf16-activation
optimization: simt, weight-streaming, hardware-fp4-conversion
status: historical-snapshot
source: FlashRT
```
