# NVIDIA Jetson Thor / SM110 Hardware Specifications

This page is the hardware-fact entry point for NVIDIA Jetson Thor. This wiki
uses the **SM110** scope for target isolation; the NVIDIA Jetson Thor Modules
Datasheet confirms a Blackwell GPU but does not itself state a compute
capability, PTX target, or the exact `sm_110*` target spelling. Verify those
on the deployed system before compiling or porting a kernel.

Jetson Thor is distinct from SM100/B200, SM103/B300, and SM120
GeForce/workstation GPUs. Do not transfer their product-level memory, power,
SM-count, or instruction assumptions without Thor-specific evidence.

## Canonical identifiers

| Field | Value |
|---|---|
| Vendor | NVIDIA |
| Product family | Jetson Thor series |
| Module SKUs | Jetson T5000, Jetson T4000 |
| GPU architecture scope | Blackwell Thor |
| Wiki compute-capability / SM target | SM110 |
| Datasheet statement about compute capability / PTX target | Not specified; verify on device |

## GPU and AI-compute specifications

| Specification | Jetson T5000 | Jetson T4000 |
|---|---:|---:|
| GPU | 2560-core NVIDIA Blackwell GPU | 1536-core NVIDIA Blackwell GPU |
| Tensor Cores | Fifth generation | Fifth generation |
| Multi-Instance GPU | 10 TPCs | 6 TPCs |
| GPU maximum frequency | 1.57 GHz | 1.53 GHz |
| AI performance | 2070 TFLOPS, FP4 sparse | 1200 TFLOPS, FP4 sparse |

The FP4 figures above are the Datasheet's sparse AI-performance figures. They
are not dense FP4, FP8, BF16, FP16, TF32, or FP32 GPU peaks, and must not be
used as substitutes for those roofline inputs.

The Datasheet gives CUDA-core and TPC counts but does not state SM count,
Tensor Core count, per-SM shared-memory capacity, L2 capacity, register-file
size, occupancy limits, or supported matrix-instruction shapes. Keep all of
those fields unknown until NVIDIA documentation or device inspection verifies
them.

## Memory, power, and CPU context

| Specification | Jetson T5000 | Jetson T4000 |
|---|---:|---:|
| Memory | 128 GB, 256-bit LPDDR5X | 64 GB, 256-bit LPDDR5X |
| Memory bandwidth | 273 GB/s | 273 GB/s |
| Configurable module power | 40 W–130 W | 40 W–70 W |
| CPU | 14-core Arm Neoverse-V3AE, 64-bit | 12-core Arm Neoverse-V3AE, 64-bit |
| CPU maximum frequency | 2.6 GHz | 2.6 GHz |
| CPU cache | 64 KB I-cache + 64 KB D-cache; 1 MB L2/core; 16 MB shared system L3 | Same |
| Programmable Vision Accelerator | 1× PVA v3 | 1× PVA v3 |

The 273 GB/s value is module memory bandwidth, not a guaranteed CUDA-kernel
bandwidth. CPU, PVA, video, networking, and other SoC activity may contend for
shared memory-system resources.

## Platform capabilities relevant to kernel integration

| Capability | Jetson T5000 | Jetson T4000 |
|---|---|---|
| Networking | 4× 25GbE | 3× 25GbE |
| PCIe | Up to 8 lanes, Gen5 | Up to 8 lanes, Gen5 |
| USB | Up to 3× USB 3.2 and 4× USB 2.0 | Same |
| Video encode / decode | 2× NVENC / 2× NVDEC | 1× NVENC / 1× NVDEC |
| Camera | Up to 20 via HSB; up to 6 via 16 lanes MIPI CSI-2; up to 32 using virtual channels | Same |
| Thermal hardware | Integrated thermal transfer plate with heatpipe | Same |

The Datasheet marks T4000 encode, decode, and low-speed I/O specifications as
preliminary and subject to change.

## Kernel-development facts still requiring verification

- Exact compute capability and valid CUDA code-generation targets.
- JetPack, Jetson Linux/L4T, CUDA, driver, PTX, and cubin compatibility.
- TMA, TMEM, tcgen05, cluster, PDL, and low-precision instruction support.
- Per-SM resources, cache hierarchy, shared-memory carveout, and occupancy
  limits.
- Dense and sparse FP32/TF32/FP16/BF16/FP8/FP4 GPU throughput.
- Nsight Compute/Nsight Systems support, metric availability, clock controls,
  and reproducible power/thermal settings.

## Sources

- NVIDIA, [Jetson Thor Modules Datasheet, 4767587, January 2026](https://nvdam.widen.net/s/mdn8tjqrzn/robotics-datasheet-update-jetson-thor-modules-nvidia-us-4767587), pages 1–3.
- NVIDIA, [Jetson Thor product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/).
- The provided signed NVIDIA download URL resolves to the same module Datasheet;
  do not copy its temporary token into durable documentation.
