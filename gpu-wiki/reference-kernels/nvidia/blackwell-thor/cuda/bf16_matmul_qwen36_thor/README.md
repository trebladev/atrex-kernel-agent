# Thor M-tile BF16 Matmul (Qwen3.6 MTP FC)

> **Usability status:** `historical-snapshot`
>
> Extracted from FlashRT. Not a standalone build target.

Thor-only BF16 row-major matmul for Qwen3.6 MTP prompt-tail FC. Caches an
`M_TILE=8` activation slab in dynamic shared memory and reuses each weight row
across the eight M rows while preserving the generic kernel's FMA order.

---

| File | Description |
|------|-------------|
| [bf16_matmul_qwen36_thor.cu](bf16_matmul_qwen36_thor.cu) | M-tile kernel, opt-in SMEM probe, launch wrapper |
| [bf16_matmul_qwen36_thor.cuh](bf16_matmul_qwen36_thor.cuh) | Public API and hardware-isolation contract |
| [NOTICE](NOTICE) | Upstream commit and Apache-2.0 provenance |

## Hardware and scope

- Hardware: Jetson Thor / SM110. Dynamic SMEM demand is
  `8 × 10240 × sizeof(BF16) = 163840` bytes.
- Must not dispatch on SM120-class devices whose per-block opt-in SMEM limit is
  below that footprint; the probe returns non-zero so callers fall back.
- Shape contract: fixed `K=10240`, BF16 input/weight/output, row-major.

## Related docs

- Optimization card:
  [sm110-large-smem-mtile-bf16-matmul.md](../../../../../docs/nvidia/blackwell-thor/kernel-opt/cuda/sm110-large-smem-mtile-bf16-matmul.md)
