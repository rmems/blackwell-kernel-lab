// L3: CUDA graph vs eager launch on RTX 5080 (sm_120). Issue #30.
// Gap: llama.cpp 9190 cannot toggle graphs (#16 cell C) — measure here.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace {

constexpr int kEmptyIters = 20000;
constexpr int kSaxpyIters = 2000;
constexpr int kChain = 32;
constexpr int kChainIters = 2000;
constexpr int kSaxpyN = 1 << 20;  // 1M floats; launch-bound, not a GEMM
constexpr int kRuns = 3;  // Run each benchmark this many times

#define BKL_CUDA(call)                                                         \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call,        \
                   cudaGetErrorString(_e));                                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void bkl_empty_kernel() {}

__global__ void bkl_saxpy_kernel(int n, float a, const float* x, float* y) {
  int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
          static_cast<int>(threadIdx.x);
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.f;
  BKL_CUDA(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

// Helper: time a callable (kernel launch body) over a stream.
// Callable is invoked once per iteration within the timing window.
// Check deferred kernel errors only after the stop event has been synchronized.
// Per-launch cudaGetLastError() adds eager-only host work to launch-bound runs.
void check_timed_launches() { BKL_CUDA(cudaGetLastError()); }

template <typename Callable>
float time_ms(cudaStream_t stream, int iters, Callable&& body) {
  cudaEvent_t start, stop;
  BKL_CUDA(cudaEventCreate(&start));
  BKL_CUDA(cudaEventCreate(&stop));
  BKL_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    body();
  }
  BKL_CUDA(cudaEventRecord(stop, stream));
  BKL_CUDA(cudaEventSynchronize(stop));
  check_timed_launches();
  float ms = elapsed_ms(start, stop);
  BKL_CUDA(cudaEventDestroy(start));
  BKL_CUDA(cudaEventDestroy(stop));
  return ms;
}

// Helper: capture, instantiate, time replays, cleanup.
// Capture callable runs once to define the graph; replay runs iters times.
template <typename CaptureCallable>
float time_graph_ms(cudaStream_t stream, int iters, CaptureCallable&& capture) {
  BKL_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  capture();
  cudaGraph_t graph = nullptr;
  BKL_CUDA(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec = nullptr;
  BKL_CUDA(cudaGraphInstantiate(&exec, graph, 0));

  // Warmup: one untimed replay.
  BKL_CUDA(cudaGraphLaunch(exec, stream));
  BKL_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  BKL_CUDA(cudaEventCreate(&start));
  BKL_CUDA(cudaEventCreate(&stop));
  BKL_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    BKL_CUDA(cudaGraphLaunch(exec, stream));
  }
  BKL_CUDA(cudaEventRecord(stop, stream));
  BKL_CUDA(cudaEventSynchronize(stop));
  check_timed_launches();
  float ms = elapsed_ms(start, stop);
  BKL_CUDA(cudaEventDestroy(start));
  BKL_CUDA(cudaEventDestroy(stop));
  BKL_CUDA(cudaGraphExecDestroy(exec));
  BKL_CUDA(cudaGraphDestroy(graph));
  return ms;
}

// Compute median from a vector of floats.
float median(std::vector<float> values) {
  if (values.empty()) return 0.0f;
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[mid - 1] + values[mid]) / 2.0f;
  }
  return values[mid];
}

// Eager launches on `stream`, already warmed.
float bench_eager_empty(cudaStream_t stream, int iters) {
  return time_ms(stream, iters, [stream]() {
    bkl_empty_kernel<<<1, 1, 0, stream>>>();
  });
}

float bench_graph_empty(cudaStream_t stream, int iters) {
  return time_graph_ms(stream, iters, [stream]() {
    bkl_empty_kernel<<<1, 1, 0, stream>>>();
  });
}

dim3 saxpy_grid() {
  constexpr int threads = 256;
  return dim3(static_cast<unsigned>((kSaxpyN + threads - 1) / threads));
}

float bench_eager_saxpy(cudaStream_t stream, int iters, float a, const float* x,
                        float* y) {
  const dim3 block(256);
  const dim3 grid = saxpy_grid();
  return time_ms(stream, iters, [=]() {
    bkl_saxpy_kernel<<<grid, block, 0, stream>>>(kSaxpyN, a, x, y);
  });
}

float bench_eager_empty_chain(cudaStream_t stream, int chain, int iters) {
  return time_ms(stream, iters, [stream, chain]() {
    for (int k = 0; k < chain; ++k) {
      bkl_empty_kernel<<<1, 1, 0, stream>>>();
    }
  });
}

float bench_graph_empty_chain(cudaStream_t stream, int chain, int iters) {
  return time_graph_ms(stream, iters, [stream, chain]() {
    for (int k = 0; k < chain; ++k) {
      bkl_empty_kernel<<<1, 1, 0, stream>>>();
    }
  });
}

float bench_graph_saxpy(cudaStream_t stream, int iters, float a, const float* x,
                        float* y) {
  const dim3 block(256);
  const dim3 grid = saxpy_grid();
  return time_graph_ms(stream, iters, [=]() {
    bkl_saxpy_kernel<<<grid, block, 0, stream>>>(kSaxpyN, a, x, y);
  });
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path out_path;
  if (argc == 3 && std::string(argv[1]) == "--out") {
    out_path = argv[2];
  } else if (argc != 1) {
    std::fprintf(stderr, "usage: %s [--out PATH]\\n", argv[0]);
    return 2;
  }
  int count = 0;
  BKL_CUDA(cudaGetDeviceCount(&count));
  if (count < 1) {
    std::fprintf(stderr, "no CUDA devices\n");
    return 1;
  }

  cudaDeviceProp prop{};
  BKL_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("device0: %s compute %d.%d\n", prop.name, prop.major, prop.minor);
  if (prop.major != 12 || prop.minor != 0) {
    std::fprintf(stderr, "expected compute 12.0 (sm_120), got %d.%d\n",
                 prop.major, prop.minor);
    return 1;
  }
  int runtime_version = 0;
  BKL_CUDA(cudaRuntimeGetVersion(&runtime_version));

  cudaStream_t stream = nullptr;
  BKL_CUDA(cudaStreamCreate(&stream));

  // Warmup (not timed).
  bkl_empty_kernel<<<1, 1, 0, stream>>>();
  BKL_CUDA(cudaGetLastError());
  BKL_CUDA(cudaStreamSynchronize(stream));

  // Run each benchmark kRuns times, report median.
  std::vector<float> eager_empty_times, graph_empty_times;
  std::vector<float> eager_chain_times, graph_chain_times;
  for (int r = 0; r < kRuns; ++r) {
    eager_empty_times.push_back(bench_eager_empty(stream, kEmptyIters));
    graph_empty_times.push_back(bench_graph_empty(stream, kEmptyIters));
    eager_chain_times.push_back(bench_eager_empty_chain(stream, kChain, kChainIters));
    graph_chain_times.push_back(bench_graph_empty_chain(stream, kChain, kChainIters));
  }
  const float eager_empty_ms = median(eager_empty_times);
  const float graph_empty_ms = median(graph_empty_times);
  const float eager_chain_ms = median(eager_chain_times);
  const float graph_chain_ms = median(graph_chain_times);

  float *x = nullptr, *y = nullptr;
  BKL_CUDA(cudaMalloc(&x, sizeof(float) * kSaxpyN));
  BKL_CUDA(cudaMalloc(&y, sizeof(float) * kSaxpyN));
  BKL_CUDA(cudaMemset(x, 0, sizeof(float) * kSaxpyN));
  BKL_CUDA(cudaMemset(y, 0, sizeof(float) * kSaxpyN));
  bkl_saxpy_kernel<<<saxpy_grid(), 256, 0, stream>>>(kSaxpyN, 1.0f, x, y);
  BKL_CUDA(cudaGetLastError());
  BKL_CUDA(cudaStreamSynchronize(stream));

  std::vector<float> eager_saxpy_times, graph_saxpy_times;
  for (int r = 0; r < kRuns; ++r) {
    eager_saxpy_times.push_back(bench_eager_saxpy(stream, kSaxpyIters, 1.0f, x, y));
    graph_saxpy_times.push_back(bench_graph_saxpy(stream, kSaxpyIters, 1.0f, x, y));
  }
  const float eager_saxpy_ms = median(eager_saxpy_times);
  const float graph_saxpy_ms = median(graph_saxpy_times);

  BKL_CUDA(cudaFree(x));
  BKL_CUDA(cudaFree(y));
  BKL_CUDA(cudaStreamDestroy(stream));

  const double eager_empty_us = 1e3 * eager_empty_ms / kEmptyIters;
  const double graph_empty_us = 1e3 * graph_empty_ms / kEmptyIters;
  const double eager_chain_us =
      1e3 * eager_chain_ms / (static_cast<double>(kChainIters) * kChain);
  const double graph_chain_us =
      1e3 * graph_chain_ms / (static_cast<double>(kChainIters) * kChain);
  const double eager_saxpy_us = 1e3 * eager_saxpy_ms / kSaxpyIters;
  const double graph_saxpy_us = 1e3 * graph_saxpy_ms / kSaxpyIters;

  std::printf("bkl graph_launch_bench sm_120\n");
  std::printf("empty iters=%d  eager_us/launch=%.4f  graph_us/launch=%.4f  speedup=%.2fx\n",
              kEmptyIters, eager_empty_us, graph_empty_us,
              graph_empty_us > 0.0 ? eager_empty_us / graph_empty_us : 0.0);
  std::printf(
      "empty_chain k=%d iters=%d  eager_us/kern=%.4f  graph_us/kern=%.4f  speedup=%.2fx\n",
      kChain, kChainIters, eager_chain_us, graph_chain_us,
      graph_chain_us > 0.0 ? eager_chain_us / graph_chain_us : 0.0);
  std::printf("saxpy n=%d iters=%d  eager_us/launch=%.4f  graph_us/launch=%.4f  speedup=%.2fx\n",
              kSaxpyN, kSaxpyIters, eager_saxpy_us, graph_saxpy_us,
              graph_saxpy_us > 0.0 ? eager_saxpy_us / graph_saxpy_us : 0.0);

  const auto now = std::chrono::system_clock::now();
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now.time_since_epoch())
                              .count();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&now_time, &utc);
  char timestamp[32];
  std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc);
  if (out_path.empty()) {
    out_path = std::filesystem::path("results") /
               ("graph-launch-bench-" + std::to_string(milliseconds) + ".json");
  }
  if (!out_path.parent_path().empty()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  FILE* out = std::fopen(out_path.c_str(), "w");
  if (out == nullptr) {
    std::perror(out_path.c_str());
    return 1;
  }
  const double empty_speedup = graph_empty_us > 0.0 ? eager_empty_us / graph_empty_us : 0.0;
  const double chain_speedup = graph_chain_us > 0.0 ? eager_chain_us / graph_chain_us : 0.0;
  const double saxpy_speedup = graph_saxpy_us > 0.0 ? eager_saxpy_us / graph_saxpy_us : 0.0;
  std::fprintf(
      out,
      "{\n  \"schema_version\": 1,\n  \"run_id\": \"graph-launch-bench-%lld\",\n"
      "  \"timestamp_utc\": \"%s\",\n"
      "  \"host\": {\"gpu_name\": \"%s\", \"vram_total_mb\": %.0f, \"cuda_runtime_version\": %d},\n"
      "  \"engine\": {\"name\": \"cuda-runtime\", \"version\": null, \"endpoint\": null},\n"
      "  \"model\": {\"id\": null, \"quant\": null, \"context_length\": null},\n"
      "  \"workload\": {\"profile\": \"l3_graph_launch_synthetic\", \"concurrency\": 1, \"steps\": null},\n"
      "  \"metrics\": {\"tool_loop_wall_ms\": [], \"tool_loop_p50_ms\": null, \"ttft_ms\": [], \"ttft_p50_ms\": null, \"tpot_ms\": [], \"tpot_p50_ms\": null, \"tokens_per_s\": null, \"prefix_cache_hit_rate\": null, \"vram_peak_mb\": null, \"vram_free_mb\": null},\n"
      "  \"benchmarks\": {\"empty\": {\"iterations\": %d, \"eager_us_per_launch\": %.4f, \"graph_us_per_launch\": %.4f, \"speedup\": %.4f}, \"empty_chain\": {\"kernels_per_graph\": %d, \"iterations\": %d, \"eager_us_per_kernel\": %.4f, \"graph_us_per_kernel\": %.4f, \"speedup\": %.4f}, \"saxpy\": {\"elements\": %d, \"iterations\": %d, \"eager_us_per_launch\": %.4f, \"graph_us_per_launch\": %.4f, \"speedup\": %.4f}},\n"
      "  \"notes\": \"Synthetic CUDA launch benchmark; not a model or decode workload. Explicit SAXPY buffers use 8 MiB.\"\n}\n",
      static_cast<long long>(milliseconds), timestamp, prop.name,
      static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0), runtime_version,
      kEmptyIters, eager_empty_us, graph_empty_us, empty_speedup, kChain, kChainIters,
      eager_chain_us, graph_chain_us, chain_speedup, kSaxpyN, kSaxpyIters,
      eager_saxpy_us, graph_saxpy_us, saxpy_speedup);
  if (std::fclose(out) != 0) {
    std::perror(out_path.c_str());
    return 1;
  }
  std::printf("wrote %s\n", out_path.c_str());
  std::printf("bkl graph_launch_bench sm_120 ok\n");
  return 0;
}
