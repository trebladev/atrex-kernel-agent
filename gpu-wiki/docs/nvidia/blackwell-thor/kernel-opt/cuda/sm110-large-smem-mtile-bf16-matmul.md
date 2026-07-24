# SM110 Large-SMEM M-tile BF16 Matmul — Quick Reference

Thor-only BF16 matmul that stages `M_TILE` activations in dynamic shared
memory and reuses each weight row across the tile. Built for Qwen3.6 MTP
prompt-tail FC (`K=10240`, `M_TILE=8`). Use when weight bandwidth dominates
and the device exposes enough opt-in shared memory.

Reference code:
[`reference-kernels/nvidia/blackwell-thor/cuda/bf16_matmul_qwen36_thor/`](../../../../../reference-kernels/nvidia/blackwell-thor/cuda/bf16_matmul_qwen36_thor/).

Upstream pin: FlashRT
`e50bec2867b0bdac1d1f4cdcda1a6e4198a8743d`.

## When to apply this recipe

- Workload: small-M / moderate-N BF16 GEMM where the same weight row is needed
  by several output rows
- Target: Jetson Thor / SM110 with
  `cudaDevAttrMaxSharedMemoryPerBlockOptin >= 163840`
- Non-target: SM120-class GPUs whose opt-in SMEM ceiling is below that
  footprint — keep a shared fallback kernel and never force this path

## Recipe (in this order)

### 1. Stage an M-tile of activations in dynamic SMEM

```text
SMEM bytes = M_TILE × K_FIXED × sizeof(BF16)
           = 8 × 10240 × 2
           = 163840
```

Load `x[m0 : m0+M_TILE, :]` cooperatively into `x_sh[]`, then synchronize.
Pad out-of-range M rows with zeros so the K loop stays uniform.

### 2. Reuse each weight row across the M tile

One warp owns one output column `n`. Read `W[n, k]` once per K step and FMA
into `acc[0..M_TILE)`. Weight traffic scales as `1 / M_TILE` versus a naive
per-row GEMV loop.

### 3. Preserve the reference FMA order

Keep the same lane coverage of K (`lane, lane+32, ...`) and the same
single-BF16 → float FMA sequence as the shared generic kernel. Bit-identical
numerics make the fast path easy to A/B against the fallback.

### 4. Probe before launch; fail closed to the shared kernel

```cpp
cudaDeviceGetAttribute(&max_optin,
    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
if (max_optin < kMtpFcSmemBytes) return 1;  // caller falls back

cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    kMtpFcSmemBytes);
```

Return non-zero on probe or launch failure so the Thor frontend can dispatch
the shared kernel explicitly. Do not silently fall back inside the Thor
specialization.

## What not to do

- Do not hard-code this path on SM120 / PRO 5000 without the opt-in probe.
- Do not change FMA order “for throughput” if you still claim bit-identical
  parity with the generic kernel.
- Do not treat Datasheet FP4 sparse AI TFLOPS as the roofline for this BF16
  kernel.

## Tags

```text
architecture: sm110
product: jetson-thor
operator: gemm
dtype: bf16
optimization: large-dynamic-smem, weight-reuse, m-tiling
status: historical-snapshot
source: FlashRT
```
