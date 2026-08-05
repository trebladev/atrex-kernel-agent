#!/usr/bin/env bash
# probe_thor.sh — collect Jetson Thor (SM110) kernel-development facts for the
# gpu-wiki SM110 hardware page and pitfalls cards.
#
# Run on the device (or in an SSH session):
#   bash probe_thor.sh                # full probe, writes thor_probe_report.md
#   REPORT=my_report.md bash probe_thor.sh
#
# Root is optional: power/clock queries degrade gracefully without it.
# The script only writes inside WORK_DIR (created next to the report).

set -u
REPORT="${REPORT:-thor_probe_report.md}"
WORK_DIR="${WORK_DIR:-.thor_probe_work}"
mkdir -p "${WORK_DIR}"

say()  { printf '%s\n' "$*"; }
emit() { printf '%s\n' "$*" | tee -a "${REPORT}"; }
section() { emit; emit "## $*"; emit; }

note_missing() { emit "- (skipped) $*"; }

: > "${REPORT}"
emit "# Jetson Thor probe report"
emit
emit "- date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
emit "- uname: $(uname -a)"

# ---------------------------------------------------------------------------
section "Toolchain"
NVCC="$(command -v nvcc || true)"
if [ -n "${NVCC}" ]; then
  emit "- nvcc: $("${NVCC}" --version | grep -i release | head -1 | sed 's/^ *//')"
else
  note_missing "nvcc not found; CUDA compile/run probes skipped"
fi
for tool in ncu nsys nvidia-smi tegrastats nvpmodel jetson_clocks gcc; do
  p="$(command -v "${tool}" 2>/dev/null || true)"
  if [ -n "${p}" ]; then emit "- ${tool}: ${p}"; else emit "- ${tool}: NOT FOUND"; fi
done

compile_probe() {
  # compile_probe <name> <arch-flag> <source-file>
  local name="$1" arch="$2" src="$3" log
  log="${WORK_DIR}/$1.build.log"
  if "${NVCC}" ${arch} -c "${src}" -o "${WORK_DIR}/$1.o" >"${log}" 2>&1; then
    emit "- ${name} (${arch}): PASS"
    return 0
  fi
  emit "- ${name} (${arch}): FAIL"
  sed 's/^/    /' "${log}" | head -5 | tee -a "${REPORT}"
  return 1
}

if [ -n "${NVCC}" ]; then
  cat > "${WORK_DIR}/hello.cu" <<'EOF'
__global__ void hello() {}
int main() { return 0; }
EOF
  section "Code-generation targets"
  compile_probe sm110_plain   "-arch=sm_110"     "${WORK_DIR}/hello.cu"
  compile_probe compute110    "-arch=compute_110" "${WORK_DIR}/hello.cu"
  compile_probe sm110a_arch   "-arch=sm_110a"    "${WORK_DIR}/hello.cu"
fi

# ---------------------------------------------------------------------------
if [ -n "${NVCC}" ]; then
  cat > "${WORK_DIR}/attrs.cu" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>

#define ATTR(name) do { \
  int v = -1; \
  if (cudaDeviceGetAttribute(&v, name, dev) == cudaSuccess) \
    printf("%s=%d\n", #name, v); \
} while (0)

int main() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) { printf("no-device\n"); return 1; }
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, dev);
  printf("name=%s\n", p.name);
  printf("compute_capability=%d.%d\n", p.major, p.minor);
  printf("sm_count=%d\n", p.multiProcessorCount);
  printf("total_global_mem_bytes=%zu\n", p.totalGlobalMem);
  printf("smem_per_block_bytes=%zu\n", p.sharedMemPerBlock);
  printf("smem_optin_per_block_bytes=%zu\n", p.sharedMemPerBlockOptin);
  printf("smem_per_sm_bytes=%zu\n", p.sharedMemPerMultiprocessor);
  printf("regs_per_block=%d\n", p.regsPerBlock);
  printf("regs_per_sm=%d\n", p.regsPerMultiprocessor);
  printf("max_threads_per_sm=%d\n", p.maxThreadsPerMultiProcessor);
  printf("warp_size=%d\n", p.warpSize);
  // CUDA 13 removed clockRate/memoryClockRate/L2CacheSize from cudaDeviceProp;
  // query them through attributes instead.
  ATTR(cudaDevAttrClockRate);
  ATTR(cudaDevAttrMemoryClockRate);
  ATTR(cudaDevAttrGlobalMemoryBusWidth);
  ATTR(cudaDevAttrL2CacheSize);
  ATTR(cudaDevAttrMaxPersistingL2CacheSize);
  ATTR(cudaDevAttrMaxBlocksPerMultiprocessor);
  ATTR(cudaDevAttrCooperativeLaunch);
  ATTR(cudaDevAttrClusterLaunch);
  ATTR(cudaDevAttrMemoryPoolsSupported);
  ATTR(cudaDevAttrUnifiedAddressing);
  ATTR(cudaDevAttrHostNativeAtomicSupported);
  ATTR(cudaDevAttrPageableMemoryAccessUsesHostPageTables);
  int clk = -1, memclk = -1, bus = -1;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
  if (memclk > 0 && bus > 0) {
    const double bw = 2.0 * (double)memclk * ((double)bus / 8.0) / 1.0e6;
    printf("theoretical_mem_bw_gbs=%.1f\n", bw);
  }
  return 0;
}
EOF
  section "Device attributes (authoritative)"
  if "${NVCC}" -arch=sm_110 "${WORK_DIR}/attrs.cu" -o "${WORK_DIR}/attrs" \
      >"${WORK_DIR}/attrs.build.log" 2>&1; then
    "${WORK_DIR}/attrs" | sed 's/^/- /' | tee -a "${REPORT}"
  else
    note_missing "attrs build failed (see ${WORK_DIR}/attrs.build.log)"
  fi
fi

# ---------------------------------------------------------------------------
if [ -n "${NVCC}" ]; then
  cat > "${WORK_DIR}/fp4_probe.cu" <<'EOF'
#include <cstdio>
#include <cuda_fp4.h>
#include <cuda_fp16.h>

__global__ void fp4_kernel(float* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  // low nibble = code 1 (0.5), high nibble = code 2 (1.0)
  const __nv_fp4x2_storage_t code = static_cast<__nv_fp4x2_storage_t>(0x21);
  const __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1);
  const __half2 h = *reinterpret_cast<const __half2*>(&raw);
  out[0] = __half2float(h.x);
  out[1] = __half2float(h.y);
#else
  out[0] = out[1] = -1.0f;
#endif
}

int main() {
  float* d = nullptr;
  cudaMalloc(&d, 2 * sizeof(float));
  fp4_kernel<<<1, 1>>>(d);
  float h[2] = {0, 0};
  cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
  cudaFree(d);
  printf("fp4_cvt_x=%.3f fp4_cvt_y=%.3f\n", h[0], h[1]);
  if (h[0] == 0.5f && h[1] == 1.0f) printf("fp4_hardware_convert=PASS\n");
  else                              printf("fp4_hardware_convert=FAIL\n");
  return 0;
}
EOF
  section "Hardware FP4 conversion (SM110 kernel card dependency)"
  if "${NVCC}" -arch=sm_110 "${WORK_DIR}/fp4_probe.cu" -o "${WORK_DIR}/fp4_probe" \
      >"${WORK_DIR}/fp4.build.log" 2>&1; then
    "${WORK_DIR}/fp4_probe" | sed 's/^/- /' | tee -a "${REPORT}"
  else
    note_missing "fp4_probe build failed (see ${WORK_DIR}/fp4.build.log)"
  fi
fi

# ---------------------------------------------------------------------------
if [ -n "${NVCC}" ]; then
  section "Blackwell feature compile probes (ptxas-validated, never launched)"
  cat > "${WORK_DIR}/tcgen05_probe.cu" <<'EOF'
__global__ void never_launched() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
#endif
}
int main() { return 0; }
EOF
  cat > "${WORK_DIR}/tma_probe.cu" <<'EOF'
__global__ void never_launched() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.commit_group;");
#endif
}
int main() { return 0; }
EOF
  cat > "${WORK_DIR}/pdl_probe.cu" <<'EOF'
#include <cuda_runtime.h>
__global__ void never_launched() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}
int main() { return 0; }
EOF
  compile_probe tcgen05_instr "-arch=sm_110a" "${WORK_DIR}/tcgen05_probe.cu"
  compile_probe bulk_tma      "-arch=sm_110"  "${WORK_DIR}/tma_probe.cu"
  compile_probe pdl_device    "-arch=sm_110"  "${WORK_DIR}/pdl_probe.cu"
fi

# ---------------------------------------------------------------------------
if [ -n "${NVCC}" ]; then
  cat > "${WORK_DIR}/bandwidth.cu" <<'EOF'
#include <algorithm>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void read_kernel(const float4* __restrict__ in, float* sink, size_t n4) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.0f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += stride)
    { const float4 v = in[i]; acc += v.x + v.y + v.z + v.w; }
  if (acc == 1234.5678f) *sink = acc;  // never true; keeps the loads alive
}

static float time_ms(void (*fn)(void*), void* arg, int iters) {
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  fn(arg);  // warmup
  cudaDeviceSynchronize();
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) fn(arg);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / iters;
}

struct CopyArgs { void* dst; const void* src; size_t bytes; };
static void do_copy(void* p) {
  const CopyArgs* a = static_cast<const CopyArgs*>(p);
  cudaMemcpy(a->dst, a->src, a->bytes, cudaMemcpyDeviceToDevice);
}
struct ReadArgs { const float4* in; float* sink; size_t n4; int grid; int block; };
static void do_read(void* p) {
  const ReadArgs* a = static_cast<const ReadArgs*>(p);
  read_kernel<<<a->grid, a->block>>>(a->in, a->sink, a->n4);
}

int main(int argc, char** argv) {
  const size_t mib = (argc > 1) ? (size_t)atoi(argv[1]) : 512;
  const size_t bytes = mib * 1024 * 1024;
  void *src = nullptr, *dst = nullptr, *sink = nullptr;
  if (cudaMalloc(&src, bytes) || cudaMalloc(&dst, bytes) || cudaMalloc(&sink, 4)) {
    printf("alloc_failed\n");
    return 1;
  }
  cudaMemset(src, 1, bytes);

  CopyArgs c{dst, src, bytes};
  const float copy_ms = time_ms(do_copy, &c, 20);
  // D2D copy moves bytes twice (read + write)
  printf("d2d_copy_gbs=%.1f\n", 2.0 * bytes / copy_ms / 1.0e6);

  int dev = 0; cudaGetDevice(&dev);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
  ReadArgs r{static_cast<const float4*>(src), static_cast<float*>(sink),
             bytes / sizeof(float4), p.multiProcessorCount * 8, 256};
  const float read_ms = time_ms(do_read, &r, 20);
  printf("kernel_read_gbs=%.1f\n", (double)bytes / read_ms / 1.0e6);

  cudaFree(src); cudaFree(dst); cudaFree(sink);
  return 0;
}
EOF
  section "Memory bandwidth (compare with the 273 GB/s module peak)"
  if "${NVCC}" -O2 -arch=sm_110 "${WORK_DIR}/bandwidth.cu" -o "${WORK_DIR}/bandwidth" \
      >"${WORK_DIR}/bandwidth.build.log" 2>&1; then
    say "running bandwidth probe ..."
    "${WORK_DIR}/bandwidth" 512 | sed 's/^/- /' | tee -a "${REPORT}"
  else
    note_missing "bandwidth build failed (see ${WORK_DIR}/bandwidth.build.log)"
  fi
fi

# ---------------------------------------------------------------------------
section "Power / clock state"
if command -v nvpmodel >/dev/null 2>&1; then
  nvpmodel -q 2>&1 | sed 's/^/    /' | tee -a "${REPORT}"
else
  note_missing "nvpmodel unavailable"
fi
if command -v jetson_clocks >/dev/null 2>&1; then
  jetson_clocks --show 2>&1 | sed 's/^/    /' | tee -a "${REPORT}"
else
  note_missing "jetson_clocks unavailable"
fi
if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 500 --logfile "${WORK_DIR}/tegrastats.log" &
  TEGRA_PID=$!
  sleep 2
  kill "${TEGRA_PID}" 2>/dev/null || true
  tail -1 "${WORK_DIR}/tegrastats.log" | sed 's/^/- tegrastats: /' | tee -a "${REPORT}"
fi

# ---------------------------------------------------------------------------
section "CPU memory contention (shared 273 GB/s fabric)"
if command -v gcc >/dev/null 2>&1 && [ -x "${WORK_DIR}/bandwidth" ]; then
  cat > "${WORK_DIR}/memhog.c" <<'EOF'
#include <stdlib.h>
#include <string.h>
int main(void) {
  const size_t n = (size_t)256 * 1024 * 1024;
  volatile char* buf = (volatile char*)malloc(n);
  if (!buf) return 1;
  for (;;) memset((void*)buf, 1, n);
  return 0;
}
EOF
  if gcc -O2 "${WORK_DIR}/memhog.c" -o "${WORK_DIR}/memhog" 2>/dev/null; then
    CORES="$(nproc 2>/dev/null || echo 4)"
    WORKERS=$(( CORES > 8 ? 8 : CORES ))
    PIDS=""
    i=0
    while [ "${i}" -lt "${WORKERS}" ]; do
      "${WORK_DIR}/memhog" & PIDS="${PIDS} $!"
      i=$((i + 1))
    done
    sleep 1
    say "running bandwidth probe under ${WORKERS} CPU memhog workers ..."
    "${WORK_DIR}/bandwidth" 512 | sed 's/^/- under-contention /' | tee -a "${REPORT}"
    kill ${PIDS} 2>/dev/null || true
    wait ${PIDS} 2>/dev/null || true
  else
    note_missing "memhog build failed"
  fi
else
  note_missing "gcc or bandwidth probe unavailable"
fi

# ---------------------------------------------------------------------------
emit
emit "Done. Attach this report (plus \`nvpmodel\` mode) to the SM110 hardware page"
emit "update or to a new pitfalls card."
say "report written to ${REPORT}"
