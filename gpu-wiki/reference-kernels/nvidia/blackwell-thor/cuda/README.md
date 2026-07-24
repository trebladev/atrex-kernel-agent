# Blackwell Thor CUDA Kernels

CUDA C++ reference kernels for Jetson Thor / SM110. Sources are extracted from
FlashRT at commit `e50bec2867b0bdac1d1f4cdcda1a6e4198a8743d` (Apache-2.0).

These trees are **historical snapshots**: they illustrate Thor-specific
optimization patterns and are not packaged as a standalone build.

---

| Directory | Description |
|-----------|-------------|
| [bf16_matmul_qwen36_thor/](bf16_matmul_qwen36_thor/) | Large dynamic-SMEM M-tile BF16 matmul with weight reuse |
| [cosmos3_reasoner_attn/](cosmos3_reasoner_attn/) | Split-KV single-query GQA decode attention (`SPLITS=24`) |
| [cosmos3_reasoner_gemv/](cosmos3_reasoner_gemv/) | M=1 W4A16 SIMT GEMV with hardware FP4 conversion |

## Related docs

- [Large-SMEM M-tile BF16 matmul](../../../../docs/nvidia/blackwell-thor/kernel-opt/cuda/sm110-large-smem-mtile-bf16-matmul.md)
- [Split-KV decode attention](../../../../docs/nvidia/blackwell-thor/kernel-opt/cuda/sm110-split-kv-decode-attention.md)
- [M=1 W4A16 SIMT GEMV](../../../../docs/nvidia/blackwell-thor/kernel-opt/cuda/sm110-m1-w4a16-simt-gemv.md)
