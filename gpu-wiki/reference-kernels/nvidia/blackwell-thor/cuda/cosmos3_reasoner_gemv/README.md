# Cosmos3 M=1 W4A16 SIMT GEMV

> **Usability status:** `historical-snapshot`
>
> Extracted from FlashRT. Compile requires `-DFLASHRT_HAVE_COSMOS3_REASONER`
> and a CUDA toolkit with `cuda_fp4.h` for Blackwell FP4 conversion.

Decode-stage `M=1` linear layer as a SIMT GEMV: one warp per output row,
packed E2M1 weights (two per byte), BF16 scale per 16 weights, BF16 activation
streamed from global memory through L1/L2 (no SMEM staging).

---

| File | Description |
|------|-------------|
| [cosmos3_reasoner_gemv.cu](cosmos3_reasoner_gemv.cu) | Warp-per-row W4A16 kernel and launch wrapper |
| [cosmos3_reasoner_gemv.cuh](cosmos3_reasoner_gemv.cuh) | Launch API |
| [NOTICE](NOTICE) | Upstream commit and Apache-2.0 provenance |

## Hardware and scope

- Hardware: Jetson Thor / SM110 (Blackwell FP4 conversion path when
  `__CUDA_ARCH__ >= 1000`).
- Workload: decode `M=1`; bottleneck is weight bandwidth, not Tensor Core tile
  occupancy.
- Layout: plain (non-swizzled) `[N, K/2]` packed weights + `[N, K/16]` BF16
  scales; `K` must be a multiple of 16.

## Related docs

- Optimization card:
  [sm110-m1-w4a16-simt-gemv.md](../../../../../docs/nvidia/blackwell-thor/kernel-opt/cuda/sm110-m1-w4a16-simt-gemv.md)
