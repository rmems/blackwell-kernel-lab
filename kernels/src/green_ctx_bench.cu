// L2: CUDA Green Context contention benchmark on RTX 5080 (sm_120). Issue #17.
// Measures a latency-sensitive kernel behind a multi-wave background kernel on
// ordinary streams versus two disjoint, dynamically queried SM partitions.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr unsigned int kBackgroundWaves = 3;
constexpr unsigned int kBlockThreads = 1024;
constexpr unsigned int kBackgroundDelayUs = 10000;
constexpr unsigned int kSensitiveIterations = 12000;
constexpr unsigned int kIndependentInvocations = 3;
constexpr unsigned int kSamplesPerMode = 7;
constexpr std::size_t kRequiredHeadroomBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12040 && \
    !defined(BKL_FORCE_NO_GREEN_CONTEXT_API)
#define BKL_HAS_GREEN_CONTEXT_API 1
#else
#define BKL_HAS_GREEN_CONTEXT_API 0
#endif

#if BKL_HAS_GREEN_CONTEXT_API
constexpr auto kReadyTimeout = std::chrono::seconds(10);
#endif

class BenchError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::string cuda_error_message(cudaError_t error, const char* expression,
                               const char* stage, const char* file, int line) {
  std::ostringstream out;
  out << stage << ": " << expression << " failed at " << file << ':' << line
      << ": " << cudaGetErrorName(error) << " (" << cudaGetErrorString(error)
      << ')';
  return out.str();
}

void cuda_check(cudaError_t error, const char* expression, const char* stage,
                const char* file, int line) {
  if (error != cudaSuccess) {
    throw BenchError(cuda_error_message(error, expression, stage, file, line));
  }
}

#define BKL_CUDA_STAGE(stage, call) \
  cuda_check((call), #call, (stage), __FILE__, __LINE__)

struct DriverErrorInfo {
  bool present = false;
  std::string stage;
  std::string name;
  std::string description;
};

#if BKL_HAS_GREEN_CONTEXT_API

std::string driver_error_name(CUresult result) {
  const char* value = nullptr;
  if (cuGetErrorName(result, &value) == CUDA_SUCCESS && value != nullptr) {
    return value;
  }
  return "CUDA_ERROR_UNKNOWN";
}

std::string driver_error_description(CUresult result) {
  const char* value = nullptr;
  if (cuGetErrorString(result, &value) == CUDA_SUCCESS && value != nullptr) {
    return value;
  }
  return "unknown CUDA Driver API error";
}

std::string driver_error_message(CUresult result, const char* expression,
                                 const char* stage, const char* file, int line) {
  std::ostringstream out;
  out << stage << ": " << expression << " failed at " << file << ':' << line
      << ": " << driver_error_name(result) << " ("
      << driver_error_description(result) << ')';
  return out.str();
}

void driver_check(CUresult result, const char* expression, const char* stage,
                  const char* file, int line) {
  if (result != CUDA_SUCCESS) {
    throw BenchError(
        driver_error_message(result, expression, stage, file, line));
  }
}

#define BKL_CU_STAGE(stage, call) \
  driver_check((call), #call, (stage), __FILE__, __LINE__)

void set_current_context(CUcontext context, const char* stage) {
  BKL_CU_STAGE(stage, cuCtxSetCurrent(context));
}

#endif

#if BKL_HAS_GREEN_CONTEXT_API

__global__ void background_delay_kernel(unsigned long long delay_cycles,
                                        int* ready_count,
                                        std::uint64_t* checksums) {
  if (threadIdx.x == 0) {
    atomicAdd(ready_count, 1);
    __threadfence_system();
  }

  std::uint64_t state =
      (static_cast<std::uint64_t>(blockIdx.x) + 1ULL) *
          0x9e3779b97f4a7c15ULL +
      static_cast<std::uint64_t>(threadIdx.x);
  const unsigned long long start = clock64();
  do {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    state *= 0x2545f4914f6cdd1dULL;
  } while (clock64() - start < delay_cycles);

  if (threadIdx.x == 0) {
    checksums[blockIdx.x] = state | 1ULL;
  }
}

__global__ void sensitive_kernel(std::uint32_t seed,
                                 std::uint32_t* output) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t state = seed ^ (index * 747796405U + 2891336453U);
  for (unsigned int iteration = 0; iteration < kSensitiveIterations;
       ++iteration) {
    state = state * 1664525U + 1013904223U;
    state ^= state >> 16;
  }
  output[index] = state;
}

std::uint32_t expected_sensitive_value(std::uint32_t seed,
                                       unsigned int index) {
  std::uint32_t state = seed ^ (index * 747796405U + 2891336453U);
  for (unsigned int iteration = 0; iteration < kSensitiveIterations;
       ++iteration) {
    state = state * 1664525U + 1013904223U;
    state ^= state >> 16;
  }
  return state;
}

double median(std::vector<double> values) {
  if (values.empty()) {
    throw BenchError("cannot compute a median from an empty distribution");
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[middle - 1] + values[middle]) / 2.0;
  }
  return values[middle];
}

#endif

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (character < 0x20U) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned int>(character) << std::dec;
        } else {
          out << static_cast<char>(character);
        }
    }
  }
  return out.str();
}

std::string json_string(const std::string& value) {
  return "\"" + json_escape(value) + "\"";
}

std::string json_number_array(const std::vector<double>& values) {
  std::ostringstream out;
  out << '[' << std::fixed << std::setprecision(6);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) out << ", ";
    out << values[index];
  }
  out << ']';
  return out.str();
}

std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&now_time, &utc);
  char timestamp[32];
  std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return timestamp;
}

std::string make_run_id() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  std::ostringstream out;
  out << "green-ctx-bench-" << milliseconds << '-' << getpid();
  return out.str();
}

struct ModeResult {
  std::vector<double> gpu_latency_ms;
  std::vector<double> host_launch_to_sync_ms;
  bool correctness = true;
};

struct InvocationResult {
  unsigned int index = 0;
  std::string order;
  ModeResult ordinary;
  ModeResult partitioned;
  double ordinary_median_ms = 0.0;
  double partitioned_median_ms = 0.0;
  double improvement_percent = 0.0;
  bool gate_pass = false;
};

struct Report {
  std::string schema = "bkl.green_ctx_bench.v1";
  std::string run_id;
  std::string timestamp;
  std::string outcome = "measured";
  std::string skipped_reason;
  DriverErrorInfo driver_error;

  std::string gpu_name;
  int compute_major = 0;
  int compute_minor = 0;
  int runtime_version = 0;
  int driver_version = 0;
  int header_version = CUDA_VERSION;
  std::size_t vram_total_bytes = 0;
  std::size_t vram_free_before_bytes = 0;
  std::size_t planned_allocation_bytes = 0;
  std::size_t projected_free_after_bytes = 0;
  std::size_t vram_free_after_allocation_bytes = 0;
  std::size_t vram_min_observed_bytes = 0;

  bool resource_query_succeeded = false;
  unsigned int sm_total = 0;
  unsigned int min_sm_partition_size = 0;
  unsigned int sm_coscheduled_alignment = 0;
  unsigned int split_min_count = 0;
  unsigned int created_groups = 0;
  unsigned int sensitive_sm_count = 0;
  unsigned int background_sm_count = 0;
  unsigned int remainder_sm_count = 0;
  int priority_high = 0;
  int priority_low = 0;

  int background_active_blocks_per_sm = 0;
  unsigned int background_grid_blocks = 0;
  unsigned long long background_delay_cycles = 0;
  std::vector<InvocationResult> invocations;
  double ordinary_overall_median_ms = 0.0;
  double partitioned_overall_median_ms = 0.0;
  double overall_improvement_percent = 0.0;
  bool follow_up_justified = false;
};

#if BKL_HAS_GREEN_CONTEXT_API

bool observe_vram_headroom(Report& report, const char* stage) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  BKL_CUDA_STAGE(stage, cudaMemGetInfo(&free_bytes, &total_bytes));
  if (report.vram_min_observed_bytes == 0 ||
      free_bytes < report.vram_min_observed_bytes) {
    report.vram_min_observed_bytes = free_bytes;
  }
  if (free_bytes < kRequiredHeadroomBytes) {
    report.outcome = "insufficient_vram_headroom";
    report.skipped_reason = "insufficient_vram_headroom";
    return false;
  }
  return true;
}

#endif

std::string nullable_unsigned(bool known, unsigned long long value) {
  return known ? std::to_string(value) : "null";
}

std::string render_report(const Report& report) {
  const bool measured = report.outcome == "measured";
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{\n"
      << "  \"schema\": " << json_string(report.schema) << ",\n"
      << "  \"schema_version\": 1,\n"
      << "  \"run_id\": " << json_string(report.run_id) << ",\n"
      << "  \"timestamp_utc\": " << json_string(report.timestamp) << ",\n"
      << "  \"outcome\": " << json_string(report.outcome) << ",\n"
      << "  \"measurement_skipped_reason\": "
      << (report.skipped_reason.empty() ? "null"
                                        : json_string(report.skipped_reason))
      << ",\n";

  if (report.driver_error.present) {
    out << "  \"driver_error\": {\"stage\": "
        << json_string(report.driver_error.stage) << ", \"name\": "
        << json_string(report.driver_error.name) << ", \"description\": "
        << json_string(report.driver_error.description) << "},\n";
  } else {
    out << "  \"driver_error\": null,\n";
  }

  out << "  \"host\": {\n"
      << "    \"gpu_name\": " << json_string(report.gpu_name) << ",\n"
      << "    \"compute_capability\": "
      << json_string(std::to_string(report.compute_major) + "." +
                     std::to_string(report.compute_minor))
      << ",\n"
      << "    \"cuda_runtime_version\": " << report.runtime_version << ",\n"
      << "    \"cuda_driver_version\": " << report.driver_version << ",\n"
      << "    \"cuda_header_version\": " << report.header_version << ",\n"
      << "    \"vram_total_bytes\": " << report.vram_total_bytes << ",\n"
      << "    \"vram_free_before_bytes\": "
      << report.vram_free_before_bytes << ",\n"
      << "    \"planned_allocation_bytes\": "
      << report.planned_allocation_bytes << ",\n"
      << "    \"projected_free_after_bytes\": "
      << report.projected_free_after_bytes << ",\n"
      << "    \"vram_free_after_allocation_bytes\": "
      << nullable_unsigned(report.vram_free_after_allocation_bytes > 0,
                           report.vram_free_after_allocation_bytes)
      << ",\n"
      << "    \"vram_min_observed_bytes\": "
      << nullable_unsigned(report.vram_min_observed_bytes > 0,
                           report.vram_min_observed_bytes)
      << ",\n"
      << "    \"required_headroom_bytes\": " << kRequiredHeadroomBytes
      << "\n  },\n";

  out << "  \"capability\": {\n"
      << "    \"compile_time_green_context_api\": "
#if BKL_HAS_GREEN_CONTEXT_API
      << "true,\n"
#else
      << "false,\n"
#endif
      << "    \"resource_query_succeeded\": "
      << (report.resource_query_succeeded ? "true" : "false") << ",\n"
      << "    \"sm_total\": "
      << nullable_unsigned(report.resource_query_succeeded, report.sm_total)
      << ",\n"
      << "    \"min_sm_partition_size\": "
      << nullable_unsigned(report.resource_query_succeeded,
                           report.min_sm_partition_size)
      << ",\n"
      << "    \"sm_coscheduled_alignment\": "
      << nullable_unsigned(report.resource_query_succeeded,
                           report.sm_coscheduled_alignment)
      << ",\n"
      << "    \"split_min_count\": "
      << nullable_unsigned(report.created_groups > 0, report.split_min_count)
      << ",\n"
      << "    \"created_groups\": " << report.created_groups << ",\n"
      << "    \"sensitive_sm_count\": "
      << nullable_unsigned(report.created_groups == 2,
                           report.sensitive_sm_count)
      << ",\n"
      << "    \"background_sm_count\": "
      << nullable_unsigned(report.created_groups == 2,
                           report.background_sm_count)
      << ",\n"
      << "    \"remainder_sm_count\": "
      << nullable_unsigned(report.created_groups == 2,
                           report.remainder_sm_count)
      << ",\n"
      << "    \"ordinary_stream_priority\": {\"sensitive\": "
      << report.priority_high << ", \"background\": " << report.priority_low
      << "},\n"
      << "    \"partitioned_stream_priority\": {\"sensitive\": "
      << report.priority_high << ", \"background\": " << report.priority_low
      << "}\n  },\n";

  out << "  \"workload\": {\n"
      << "    \"background_block_threads\": " << kBlockThreads << ",\n"
      << "    \"background_active_blocks_per_sm\": "
      << report.background_active_blocks_per_sm << ",\n"
      << "    \"background_grid_blocks\": "
      << report.background_grid_blocks << ",\n"
      << "    \"background_waves\": " << kBackgroundWaves << ",\n"
      << "    \"background_delay_us_per_block\": "
      << kBackgroundDelayUs << ",\n"
      << "    \"background_delay_cycles\": "
      << report.background_delay_cycles << ",\n"
      << "    \"sensitive_block_threads\": " << kBlockThreads << ",\n"
      << "    \"sensitive_blocks\": 1,\n"
      << "    \"sensitive_iterations\": " << kSensitiveIterations << ",\n"
      << "    \"samples_per_mode\": " << kSamplesPerMode << ",\n"
      << "    \"independent_invocations\": " << kIndependentInvocations
      << "\n  },\n";

  out << "  \"invocations\": [";
  for (std::size_t index = 0; index < report.invocations.size(); ++index) {
    const InvocationResult& invocation = report.invocations[index];
    if (index != 0) out << ',';
    out << "\n    {\n"
        << "      \"index\": " << invocation.index << ",\n"
        << "      \"order\": " << json_string(invocation.order) << ",\n"
        << "      \"ordinary_gpu_latency_ms\": "
        << json_number_array(invocation.ordinary.gpu_latency_ms) << ",\n"
        << "      \"partitioned_gpu_latency_ms\": "
        << json_number_array(invocation.partitioned.gpu_latency_ms) << ",\n"
        << "      \"ordinary_host_launch_to_sync_ms\": "
        << json_number_array(invocation.ordinary.host_launch_to_sync_ms)
        << ",\n"
        << "      \"partitioned_host_launch_to_sync_ms\": "
        << json_number_array(invocation.partitioned.host_launch_to_sync_ms)
        << ",\n"
        << "      \"ordinary_median_ms\": "
        << invocation.ordinary_median_ms << ",\n"
        << "      \"partitioned_median_ms\": "
        << invocation.partitioned_median_ms << ",\n"
        << "      \"improvement_percent\": "
        << invocation.improvement_percent << ",\n"
        << "      \"correctness\": "
        << ((invocation.ordinary.correctness &&
             invocation.partitioned.correctness)
                ? "true"
                : "false")
        << ",\n"
        << "      \"gate_pass\": "
        << (invocation.gate_pass ? "true" : "false") << "\n    }";
  }
  if (!report.invocations.empty()) out << '\n' << "  ";
  out << "],\n";

  out << "  \"summary\": {\n"
      << "    \"ordinary_median_ms\": ";
  if (measured) {
    out << report.ordinary_overall_median_ms;
  } else {
    out << "null";
  }
  out << ",\n    \"partitioned_median_ms\": ";
  if (measured) {
    out << report.partitioned_overall_median_ms;
  } else {
    out << "null";
  }
  out << ",\n    \"improvement_percent\": ";
  if (measured) {
    out << report.overall_improvement_percent;
  } else {
    out << "null";
  }
  out << "\n  },\n";

  out << "  \"decision\": {\n"
      << "    \"required_improvement_percent\": 10.000000,\n"
      << "    \"passed_all_three\": "
      << (measured ? (report.follow_up_justified ? "true" : "false")
                   : "null")
      << ",\n"
      << "    \"follow_up_justified\": "
      << (report.follow_up_justified ? "true" : "false") << "\n  },\n"
      << "  \"notes\": "
      << json_string(
             "First-party compute-only L2 scheduling benchmark; not MIG, a "
             "model-serving benchmark, or proof of vLLM behavior.")
      << "\n}\n";
  return out.str();
}

void write_report_atomically(const std::filesystem::path& output_path,
                             const Report& report) {
  if (output_path.empty()) {
    throw BenchError("output path must not be empty");
  }
  if (!output_path.parent_path().empty()) {
    std::error_code directory_error;
    std::filesystem::create_directories(output_path.parent_path(),
                                        directory_error);
    if (directory_error) {
      throw BenchError("could not create output directory: " +
                       directory_error.message());
    }
  }

  const std::filesystem::path temporary_path =
      output_path.string() + ".tmp-" + std::to_string(getpid());
  {
    std::ofstream output(temporary_path,
                         std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output) {
      throw BenchError("could not open temporary output " +
                       temporary_path.string());
    }
    output << render_report(report);
    output.flush();
    if (!output) {
      output.close();
      std::error_code ignored;
      std::filesystem::remove(temporary_path, ignored);
      throw BenchError("could not write complete JSON output " +
                       temporary_path.string());
    }
  }

  std::error_code rename_error;
  std::filesystem::rename(temporary_path, output_path, rename_error);
  if (rename_error) {
    std::error_code ignored;
    std::filesystem::remove(temporary_path, ignored);
    throw BenchError("could not publish JSON output " + output_path.string() +
                     ": " + rename_error.message());
  }
}

#if BKL_HAS_GREEN_CONTEXT_API

class Buffers {
 public:
  Buffers() = default;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  ~Buffers() {
    if (sensitive_device_ != nullptr) cudaFree(sensitive_device_);
    if (background_device_ != nullptr) cudaFree(background_device_);
    if (ready_host_ != nullptr) cudaFreeHost(ready_host_);
  }

  void allocate(unsigned int background_blocks) {
    background_blocks_ = background_blocks;
    BKL_CUDA_STAGE("allocate sensitive output",
                   cudaMalloc(&sensitive_device_,
                              kBlockThreads * sizeof(std::uint32_t)));
    BKL_CUDA_STAGE("allocate background checksums",
                   cudaMalloc(&background_device_,
                              static_cast<std::size_t>(background_blocks_) *
                                  sizeof(std::uint64_t)));
    BKL_CUDA_STAGE("allocate mapped readiness counter",
                   cudaHostAlloc(&ready_host_, sizeof(int), cudaHostAllocMapped));
    BKL_CUDA_STAGE("map readiness counter",
                   cudaHostGetDevicePointer(&ready_device_, ready_host_, 0));
  }

  void release() {
    if (sensitive_device_ != nullptr) {
      BKL_CUDA_STAGE("free sensitive output", cudaFree(sensitive_device_));
      sensitive_device_ = nullptr;
    }
    if (background_device_ != nullptr) {
      BKL_CUDA_STAGE("free background checksums", cudaFree(background_device_));
      background_device_ = nullptr;
    }
    if (ready_host_ != nullptr) {
      BKL_CUDA_STAGE("free mapped readiness counter", cudaFreeHost(ready_host_));
      ready_host_ = nullptr;
      ready_device_ = nullptr;
    }
  }

  std::uint32_t* sensitive_device() const { return sensitive_device_; }
  std::uint64_t* background_device() const { return background_device_; }
  int* ready_host() const { return ready_host_; }
  int* ready_device() const { return ready_device_; }
  unsigned int background_blocks() const { return background_blocks_; }

 private:
  std::uint32_t* sensitive_device_ = nullptr;
  std::uint64_t* background_device_ = nullptr;
  int* ready_host_ = nullptr;
  int* ready_device_ = nullptr;
  unsigned int background_blocks_ = 0;
};

void reset_sample_buffers(Buffers& buffers) {
  *buffers.ready_host() = 0;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  BKL_CUDA_STAGE("reset sensitive output",
                 cudaMemset(buffers.sensitive_device(), 0,
                            kBlockThreads * sizeof(std::uint32_t)));
  BKL_CUDA_STAGE(
      "reset background checksums",
      cudaMemset(buffers.background_device(), 0,
                 static_cast<std::size_t>(buffers.background_blocks()) *
                     sizeof(std::uint64_t)));
}

void wait_for_background_ready(const int* ready_host, int required) {
  const auto deadline = std::chrono::steady_clock::now() + kReadyTimeout;
  while (*reinterpret_cast<volatile const int*>(ready_host) < required) {
    if (std::chrono::steady_clock::now() >= deadline) {
      std::ostringstream message;
      message << "background readiness timed out: observed " << *ready_host
              << ", required " << required;
      throw BenchError(message.str());
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

void validate_sensitive_output(Buffers& buffers, std::uint32_t seed) {
  std::vector<std::uint32_t> output(kBlockThreads);
  BKL_CUDA_STAGE("copy sensitive output",
                 cudaMemcpy(output.data(), buffers.sensitive_device(),
                            output.size() * sizeof(output[0]),
                            cudaMemcpyDeviceToHost));
  for (unsigned int index = 0; index < output.size(); ++index) {
    const std::uint32_t expected = expected_sensitive_value(seed, index);
    if (output[index] != expected) {
      std::ostringstream message;
      message << "sensitive output mismatch at index " << index << ": got "
              << output[index] << ", expected " << expected;
      throw BenchError(message.str());
    }
  }
}

void validate_background_output(Buffers& buffers) {
  std::vector<std::uint64_t> output(buffers.background_blocks());
  BKL_CUDA_STAGE("copy background checksums",
                 cudaMemcpy(output.data(), buffers.background_device(),
                            output.size() * sizeof(output[0]),
                            cudaMemcpyDeviceToHost));
  const auto missing =
      std::find(output.begin(), output.end(), static_cast<std::uint64_t>(0));
  if (missing != output.end()) {
    throw BenchError("background checksum validation found an unexecuted block");
  }
}

struct SampleResult {
  double gpu_latency_ms = 0.0;
  double host_launch_to_sync_ms = 0.0;
};

SampleResult run_sample(cudaStream_t sensitive_stream,
                        cudaStream_t background_stream,
                        CUcontext sensitive_context,
                        CUcontext background_context,
                        CUcontext primary_context,
                        int background_ready_target,
                        unsigned long long background_delay_cycles,
                        std::uint32_t seed, Buffers& buffers) {
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  try {
    set_current_context(primary_context,
                        "restore primary context before sample reset");
    reset_sample_buffers(buffers);

    set_current_context(background_context,
                        "select background execution context");
    background_delay_kernel<<<buffers.background_blocks(), kBlockThreads, 0,
                              background_stream>>>(
        background_delay_cycles, buffers.ready_device(),
        buffers.background_device());
    BKL_CUDA_STAGE("launch background kernel", cudaGetLastError());
    wait_for_background_ready(buffers.ready_host(), background_ready_target);

    set_current_context(sensitive_context,
                        "select sensitive execution context");
    BKL_CUDA_STAGE("create sensitive start event",
                   cudaEventCreate(&start_event));
    BKL_CUDA_STAGE("create sensitive stop event", cudaEventCreate(&stop_event));
    BKL_CUDA_STAGE("record sensitive start event",
                   cudaEventRecord(start_event, sensitive_stream));
    const auto host_start = std::chrono::steady_clock::now();
    sensitive_kernel<<<1, kBlockThreads, 0, sensitive_stream>>>(
        seed, buffers.sensitive_device());
    BKL_CUDA_STAGE("launch sensitive kernel", cudaGetLastError());
    BKL_CUDA_STAGE("record sensitive stop event",
                   cudaEventRecord(stop_event, sensitive_stream));
    BKL_CUDA_STAGE("synchronize sensitive stop event",
                   cudaEventSynchronize(stop_event));
    const auto host_stop = std::chrono::steady_clock::now();

    float gpu_ms = 0.0F;
    BKL_CUDA_STAGE("read sensitive event latency",
                   cudaEventElapsedTime(&gpu_ms, start_event, stop_event));

    BKL_CUDA_STAGE("destroy sensitive stop event",
                   cudaEventDestroy(stop_event));
    stop_event = nullptr;
    BKL_CUDA_STAGE("destroy sensitive start event",
                   cudaEventDestroy(start_event));
    start_event = nullptr;

    set_current_context(background_context,
                        "select background context for synchronization");
    BKL_CUDA_STAGE("synchronize background stream",
                   cudaStreamSynchronize(background_stream));
    set_current_context(primary_context,
                        "restore primary context after sample");
    validate_sensitive_output(buffers, seed);
    validate_background_output(buffers);

    const double host_ms =
        std::chrono::duration<double, std::milli>(host_stop - host_start)
            .count();
    return {static_cast<double>(gpu_ms), host_ms};
  } catch (...) {
    cuCtxSetCurrent(sensitive_context);
    if (stop_event != nullptr) {
      cudaEventDestroy(stop_event);
      stop_event = nullptr;
    }
    if (start_event != nullptr) {
      cudaEventDestroy(start_event);
      start_event = nullptr;
    }
    cuCtxSetCurrent(background_context);
    cudaStreamSynchronize(background_stream);
    cuCtxSetCurrent(primary_context);
    throw;
  }
}

ModeResult run_mode(cudaStream_t sensitive_stream,
                    cudaStream_t background_stream,
                    CUcontext sensitive_context,
                    CUcontext background_context,
                    CUcontext primary_context,
                    int background_ready_target,
                    unsigned long long background_delay_cycles,
                    unsigned int invocation_index, Buffers& buffers) {
  // One complete overlap warmup is intentionally excluded from distributions.
  const std::uint32_t warmup_seed = 0x6b8b4567U ^ invocation_index;
  static_cast<void>(run_sample(
      sensitive_stream, background_stream, sensitive_context,
      background_context, primary_context, background_ready_target,
      background_delay_cycles, warmup_seed, buffers));

  ModeResult result;
  result.gpu_latency_ms.reserve(kSamplesPerMode);
  result.host_launch_to_sync_ms.reserve(kSamplesPerMode);
  for (unsigned int sample = 0; sample < kSamplesPerMode; ++sample) {
    const std::uint32_t seed =
        0x9e3779b9U ^ (invocation_index * 131U + sample * 17U);
    const SampleResult sample_result =
        run_sample(sensitive_stream, background_stream, sensitive_context,
                   background_context, primary_context,
                   background_ready_target, background_delay_cycles, seed,
                   buffers);
    result.gpu_latency_ms.push_back(sample_result.gpu_latency_ms);
    result.host_launch_to_sync_ms.push_back(
        sample_result.host_launch_to_sync_ms);
  }
  return result;
}

class OrdinaryStreams {
 public:
  OrdinaryStreams(int high_priority, int low_priority) {
    BKL_CUDA_STAGE(
        "create ordinary sensitive stream",
        cudaStreamCreateWithPriority(&sensitive_, cudaStreamNonBlocking,
                                     high_priority));
    try {
      BKL_CUDA_STAGE(
          "create ordinary background stream",
          cudaStreamCreateWithPriority(&background_, cudaStreamNonBlocking,
                                       low_priority));
    } catch (...) {
      cudaStreamDestroy(sensitive_);
      sensitive_ = nullptr;
      throw;
    }
  }

  OrdinaryStreams(const OrdinaryStreams&) = delete;
  OrdinaryStreams& operator=(const OrdinaryStreams&) = delete;

  ~OrdinaryStreams() {
    if (background_ != nullptr) cudaStreamDestroy(background_);
    if (sensitive_ != nullptr) cudaStreamDestroy(sensitive_);
  }

  void close() {
    if (background_ != nullptr) {
      BKL_CUDA_STAGE("destroy ordinary background stream",
                     cudaStreamDestroy(background_));
      background_ = nullptr;
    }
    if (sensitive_ != nullptr) {
      BKL_CUDA_STAGE("destroy ordinary sensitive stream",
                     cudaStreamDestroy(sensitive_));
      sensitive_ = nullptr;
    }
  }

  cudaStream_t sensitive() const { return sensitive_; }
  cudaStream_t background() const { return background_; }

 private:
  cudaStream_t sensitive_ = nullptr;
  cudaStream_t background_ = nullptr;
};

struct Partition {
  CUdevResource groups[2]{};
  CUdevResource remainder{};
  unsigned int group_count = 0;
  unsigned int min_count = 0;
};

bool expected_driver_gate(CUresult result, const std::string& stage,
                          Report& report) {
  if (result != CUDA_ERROR_NOT_SUPPORTED &&
      result != CUDA_ERROR_NOT_PERMITTED) {
    return false;
  }
  report.outcome = result == CUDA_ERROR_NOT_SUPPORTED ? "unsupported"
                                                       : "not_permitted";
  report.skipped_reason = result == CUDA_ERROR_NOT_SUPPORTED
                              ? "cuda_error_not_supported"
                              : "cuda_error_not_permitted";
  report.driver_error.present = true;
  report.driver_error.stage = stage;
  report.driver_error.name = driver_error_name(result);
  report.driver_error.description = driver_error_description(result);
  return true;
}

bool no_legal_split_result(CUresult result, const std::string& stage,
                           Report& report) {
  if (result != CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION) {
    return false;
  }
  report.outcome = "no_legal_split";
  report.skipped_reason = "no_legal_two_way_sm_split";
  report.driver_error.present = true;
  report.driver_error.stage = stage;
  report.driver_error.name = driver_error_name(result);
  report.driver_error.description = driver_error_description(result);
  return true;
}

unsigned int aligned_split_count(const CUdevResource& resource) {
  const unsigned int alignment = resource.sm.smCoscheduledAlignment;
  const unsigned int minimum = resource.sm.minSmPartitionSize;
  if (alignment == 0 || minimum == 0 || resource.sm.smCount < 2 * minimum) {
    return 0;
  }
  const unsigned int half = resource.sm.smCount / 2;
  const unsigned int candidate = (half / alignment) * alignment;
  const unsigned int minimum_aligned =
      ((minimum + alignment - 1) / alignment) * alignment;
  if (candidate < minimum_aligned || candidate == 0 ||
      candidate > resource.sm.smCount / 2) {
    return 0;
  }
  return candidate;
}

bool query_partition(CUdevice device, Report& report, Partition& partition,
                     bool set_report_fields) {
  CUdevResource full_resource{};
  const CUresult query_result = cuDeviceGetDevResource(
      device, &full_resource, CU_DEV_RESOURCE_TYPE_SM);
  if (expected_driver_gate(query_result, "query_sm_resource", report)) {
    return false;
  }
  BKL_CU_STAGE("query SM resource", query_result);
  if (full_resource.type != CU_DEV_RESOURCE_TYPE_SM ||
      full_resource.sm.smCount == 0 ||
      full_resource.sm.minSmPartitionSize == 0 ||
      full_resource.sm.smCoscheduledAlignment == 0) {
    throw BenchError("SM resource query returned invalid capability fields");
  }

  if (set_report_fields) {
    report.resource_query_succeeded = true;
    report.sm_total = full_resource.sm.smCount;
    report.min_sm_partition_size = full_resource.sm.minSmPartitionSize;
    report.sm_coscheduled_alignment =
        full_resource.sm.smCoscheduledAlignment;
  } else if (report.sm_total != full_resource.sm.smCount ||
             report.min_sm_partition_size !=
                 full_resource.sm.minSmPartitionSize ||
             report.sm_coscheduled_alignment !=
                 full_resource.sm.smCoscheduledAlignment) {
    throw BenchError("SM resource capability changed during one benchmark run");
  }

  partition.min_count = aligned_split_count(full_resource);
  if (partition.min_count == 0) {
    report.outcome = "no_legal_split";
    report.skipped_reason = "no_legal_two_way_sm_split";
    return false;
  }

  unsigned int discovered_groups = 2;
  const CUresult discovery_result = cuDevSmResourceSplitByCount(
      nullptr, &discovered_groups, &full_resource, nullptr, 0,
      partition.min_count);
  if (expected_driver_gate(discovery_result, "discover_sm_split", report) ||
      no_legal_split_result(discovery_result, "discover_sm_split", report)) {
    return false;
  }
  BKL_CU_STAGE("discover two-way SM split", discovery_result);
  if (discovered_groups < 2) {
    report.outcome = "no_legal_split";
    report.skipped_reason = "no_legal_two_way_sm_split";
    return false;
  }

  partition.group_count = 2;
  const CUresult split_result = cuDevSmResourceSplitByCount(
      partition.groups, &partition.group_count, &full_resource,
      &partition.remainder, 0, partition.min_count);
  if (expected_driver_gate(split_result, "create_sm_split", report) ||
      no_legal_split_result(split_result, "create_sm_split", report)) {
    return false;
  }
  BKL_CU_STAGE("create two-way SM split", split_result);
  if (partition.group_count != 2 ||
      partition.groups[0].type != CU_DEV_RESOURCE_TYPE_SM ||
      partition.groups[1].type != CU_DEV_RESOURCE_TYPE_SM ||
      partition.groups[0].sm.smCount == 0 ||
      partition.groups[1].sm.smCount == 0) {
    report.outcome = "no_legal_split";
    report.skipped_reason = "no_legal_two_way_sm_split";
    return false;
  }

  if (set_report_fields) {
    report.split_min_count = partition.min_count;
    report.created_groups = partition.group_count;
    report.sensitive_sm_count = partition.groups[0].sm.smCount;
    report.background_sm_count = partition.groups[1].sm.smCount;
    report.remainder_sm_count =
        partition.remainder.type == CU_DEV_RESOURCE_TYPE_SM
            ? partition.remainder.sm.smCount
            : 0;
  } else if (partition.min_count != report.split_min_count ||
             partition.groups[0].sm.smCount != report.sensitive_sm_count ||
             partition.groups[1].sm.smCount != report.background_sm_count) {
    throw BenchError("SM split changed during one benchmark run");
  }
  return true;
}

class GreenPair {
 public:
  GreenPair() = default;
  GreenPair(const GreenPair&) = delete;
  GreenPair& operator=(const GreenPair&) = delete;

  ~GreenPair() { cleanup_best_effort(); }

  bool create(CUdevice device, Partition& partition, int high_priority,
              int low_priority, Report& report) {
    CUdevResourceDesc descriptors[2] = {nullptr, nullptr};
    for (unsigned int index = 0; index < 2; ++index) {
      const CUresult descriptor_result = cuDevResourceGenerateDesc(
          &descriptors[index], &partition.groups[index], 1);
      if (expected_driver_gate(descriptor_result, "generate_resource_desc",
                               report)) {
        return false;
      }
      BKL_CU_STAGE("generate Green Context resource descriptor",
                   descriptor_result);
    }

    for (unsigned int index = 0; index < 2; ++index) {
      const CUresult context_result = cuGreenCtxCreate(
          &contexts_[index], descriptors[index], device, CU_GREEN_CTX_NONE);
      if (expected_driver_gate(context_result, "create_green_context",
                               report)) {
        cleanup_checked();
        return false;
      }
      BKL_CU_STAGE("create Green Context", context_result);
      BKL_CU_STAGE("convert Green Context to execution context",
                   cuCtxFromGreenCtx(&execution_contexts_[index],
                                     contexts_[index]));
    }

    const int priorities[2] = {high_priority, low_priority};
    for (unsigned int index = 0; index < 2; ++index) {
      const CUresult stream_result = cuGreenCtxStreamCreate(
          &streams_[index], contexts_[index], CU_STREAM_NON_BLOCKING,
          priorities[index]);
      if (expected_driver_gate(stream_result, "create_green_stream", report)) {
        cleanup_checked();
        return false;
      }
      BKL_CU_STAGE("create Green Context stream", stream_result);
    }
    return true;
  }

  void close() { cleanup_checked(); }

  cudaStream_t sensitive() const { return streams_[0]; }
  cudaStream_t background() const { return streams_[1]; }
  CUcontext sensitive_context() const { return execution_contexts_[0]; }
  CUcontext background_context() const { return execution_contexts_[1]; }

 private:
  void cleanup_checked() {
    for (int index = 1; index >= 0; --index) {
      if (streams_[index] != nullptr) {
        BKL_CU_STAGE("destroy Green Context stream",
                     cuStreamDestroy(streams_[index]));
        streams_[index] = nullptr;
      }
    }
    for (int index = 1; index >= 0; --index) {
      if (contexts_[index] != nullptr) {
        BKL_CU_STAGE("destroy Green Context",
                     cuGreenCtxDestroy(contexts_[index]));
        contexts_[index] = nullptr;
        execution_contexts_[index] = nullptr;
      }
    }
  }

  void cleanup_best_effort() noexcept {
    for (int index = 1; index >= 0; --index) {
      if (streams_[index] != nullptr) {
        cuStreamDestroy(streams_[index]);
        streams_[index] = nullptr;
      }
    }
    for (int index = 1; index >= 0; --index) {
      if (contexts_[index] != nullptr) {
        cuGreenCtxDestroy(contexts_[index]);
        contexts_[index] = nullptr;
        execution_contexts_[index] = nullptr;
      }
    }
  }

  CUgreenCtx contexts_[2] = {nullptr, nullptr};
  CUcontext execution_contexts_[2] = {nullptr, nullptr};
  CUstream streams_[2] = {nullptr, nullptr};
};

#endif

void print_summary(const Report& report,
                   const std::filesystem::path& output_path) {
  std::printf("bkl green_ctx_bench sm_120\n");
  std::printf("outcome=%s sm_total=%u min_partition=%u alignment=%u\n",
              report.outcome.c_str(), report.sm_total,
              report.min_sm_partition_size,
              report.sm_coscheduled_alignment);
  if (report.outcome == "measured") {
    std::printf("split sensitive=%u background=%u remainder=%u\n",
                report.sensitive_sm_count, report.background_sm_count,
                report.remainder_sm_count);
    for (const InvocationResult& invocation : report.invocations) {
      std::printf(
          "invocation=%u order=%s ordinary_median_ms=%.4f "
          "partitioned_median_ms=%.4f improvement=%.2f%% gate=%s\n",
          invocation.index, invocation.order.c_str(),
          invocation.ordinary_median_ms, invocation.partitioned_median_ms,
          invocation.improvement_percent,
          invocation.gate_pass ? "pass" : "fail");
    }
    std::printf("follow_up_justified=%s\n",
                report.follow_up_justified ? "true" : "false");
  } else {
    std::printf("measurement_skipped_reason=%s\n",
                report.skipped_reason.c_str());
  }
  std::printf("wrote %s\n", output_path.c_str());
  std::printf("bkl green_ctx_bench sm_120 ok\n");
}

int run_benchmark(const std::filesystem::path& requested_output) {
  Report report;
  report.run_id = make_run_id();
  report.timestamp = utc_timestamp();

  int device_count = 0;
  BKL_CUDA_STAGE("query CUDA device count", cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    throw BenchError("no CUDA devices are visible");
  }
  BKL_CUDA_STAGE("select CUDA device 0", cudaSetDevice(0));
  BKL_CUDA_STAGE("initialize CUDA primary context", cudaFree(nullptr));

  cudaDeviceProp properties{};
  BKL_CUDA_STAGE("query CUDA device properties",
                 cudaGetDeviceProperties(&properties, 0));
  report.gpu_name = properties.name;
  report.compute_major = properties.major;
  report.compute_minor = properties.minor;
  report.vram_total_bytes = properties.totalGlobalMem;
  if (properties.major != 12 || properties.minor != 0) {
    std::ostringstream message;
    message << "expected compute 12.0 (sm_120), got " << properties.major
            << '.' << properties.minor;
    throw BenchError(message.str());
  }
  BKL_CUDA_STAGE("query CUDA runtime version",
                 cudaRuntimeGetVersion(&report.runtime_version));
  BKL_CUDA_STAGE("query CUDA driver version",
                 cudaDriverGetVersion(&report.driver_version));
  BKL_CUDA_STAGE("query stream priority range",
                 cudaDeviceGetStreamPriorityRange(&report.priority_low,
                                                  &report.priority_high));

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  BKL_CUDA_STAGE("query free VRAM",
                 cudaMemGetInfo(&free_bytes, &total_bytes));
  report.vram_free_before_bytes = free_bytes;
  report.vram_total_bytes = total_bytes;

  const std::filesystem::path output_path =
      requested_output.empty()
          ? std::filesystem::path("results") /
                (report.run_id + std::string(".json"))
          : requested_output;

#if !BKL_HAS_GREEN_CONTEXT_API
  report.outcome = "compile_time_api_unavailable";
  report.skipped_reason = "cuda_headers_precede_green_context_driver_api";
  write_report_atomically(output_path, report);
  print_summary(report, output_path);
  return 0;
#else
  BKL_CU_STAGE("initialize CUDA Driver API", cuInit(0));
  CUdevice driver_device = 0;
  BKL_CU_STAGE("get CUDA Driver device 0", cuDeviceGet(&driver_device, 0));
  CUcontext primary_context = nullptr;
  BKL_CU_STAGE("get CUDA primary context",
               cuCtxGetCurrent(&primary_context));
  if (primary_context == nullptr) {
    throw BenchError("CUDA Runtime did not leave a primary context current");
  }

  Partition initial_partition;
  if (!query_partition(driver_device, report, initial_partition, true)) {
    write_report_atomically(output_path, report);
    print_summary(report, output_path);
    return 0;
  }
  if (report.sm_total !=
      static_cast<unsigned int>(properties.multiProcessorCount)) {
    throw BenchError(
        "Runtime and Driver APIs disagree on the device SM count");
  }

  BKL_CUDA_STAGE(
      "query background kernel occupancy",
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &report.background_active_blocks_per_sm, background_delay_kernel,
          kBlockThreads, 0));
  if (report.background_active_blocks_per_sm <= 0) {
    throw BenchError("background kernel has zero active blocks per SM");
  }
  const unsigned long long background_blocks =
      static_cast<unsigned long long>(report.background_active_blocks_per_sm) *
      report.sm_total * kBackgroundWaves;
  if (background_blocks > std::numeric_limits<unsigned int>::max()) {
    throw BenchError("background grid exceeds CUDA grid dimension range");
  }
  report.background_grid_blocks =
      static_cast<unsigned int>(background_blocks);
  int clock_rate_khz = 0;
  BKL_CUDA_STAGE("query device clock rate",
                 cudaDeviceGetAttribute(&clock_rate_khz,
                                        cudaDevAttrClockRate, 0));
  if (clock_rate_khz <= 0) {
    throw BenchError("device reported a non-positive clock rate");
  }
  report.background_delay_cycles =
      static_cast<unsigned long long>(clock_rate_khz) *
      kBackgroundDelayUs / 1000ULL;
  if (report.background_delay_cycles == 0) {
    throw BenchError("derived background delay is zero cycles");
  }

  const std::size_t sensitive_bytes =
      kBlockThreads * sizeof(std::uint32_t);
  const std::size_t background_bytes =
      static_cast<std::size_t>(report.background_grid_blocks) *
      sizeof(std::uint64_t);
  if (background_bytes >
      std::numeric_limits<std::size_t>::max() - sensitive_bytes) {
    throw BenchError("planned allocation byte count overflow");
  }
  report.planned_allocation_bytes = sensitive_bytes + background_bytes;
  report.projected_free_after_bytes =
      free_bytes >= report.planned_allocation_bytes
          ? free_bytes - report.planned_allocation_bytes
          : 0;
  if (free_bytes < report.planned_allocation_bytes ||
      report.projected_free_after_bytes < kRequiredHeadroomBytes) {
    report.outcome = "insufficient_vram_headroom";
    report.skipped_reason = "insufficient_vram_headroom";
    write_report_atomically(output_path, report);
    print_summary(report, output_path);
    return 0;
  }

  Buffers buffers;
  buffers.allocate(report.background_grid_blocks);
  {
    std::size_t free_after_allocation = 0;
    std::size_t total_after_allocation = 0;
    BKL_CUDA_STAGE("query VRAM after benchmark allocation",
                   cudaMemGetInfo(&free_after_allocation,
                                  &total_after_allocation));
    report.vram_free_after_allocation_bytes = free_after_allocation;
    report.vram_min_observed_bytes = free_after_allocation;
    if (free_after_allocation < kRequiredHeadroomBytes) {
      report.outcome = "insufficient_vram_headroom";
      report.skipped_reason = "insufficient_vram_headroom";
      buffers.release();
      write_report_atomically(output_path, report);
      print_summary(report, output_path);
      return 0;
    }
  }

  auto run_ordinary = [&](unsigned int invocation_index) {
    OrdinaryStreams streams(report.priority_high, report.priority_low);
    const int ready_target =
        report.background_active_blocks_per_sm *
        static_cast<int>(report.sm_total);
    ModeResult result =
        run_mode(streams.sensitive(), streams.background(), primary_context,
                 primary_context, primary_context, ready_target,
                 report.background_delay_cycles, invocation_index, buffers);
    streams.close();
    static_cast<void>(
        observe_vram_headroom(report, "observe ordinary-mode VRAM"));
    return result;
  };

  auto run_partitioned = [&](unsigned int invocation_index) {
    Partition partition;
    if (!query_partition(driver_device, report, partition, false)) {
      return ModeResult{};
    }
    GreenPair pair;
    if (!pair.create(driver_device, partition, report.priority_high,
                     report.priority_low, report)) {
      return ModeResult{};
    }
    if (!observe_vram_headroom(report,
                               "observe Green Context creation VRAM")) {
      pair.close();
      return ModeResult{};
    }
    const int ready_target =
        report.background_active_blocks_per_sm *
        static_cast<int>(report.background_sm_count);
    ModeResult result =
        run_mode(pair.sensitive(), pair.background(), pair.sensitive_context(),
                 pair.background_context(), primary_context, ready_target,
                 report.background_delay_cycles, invocation_index, buffers);
    static_cast<void>(
        observe_vram_headroom(report, "observe partitioned-mode VRAM"));
    pair.close();
    return result;
  };

  for (unsigned int invocation_index = 0;
       invocation_index < kIndependentInvocations; ++invocation_index) {
    InvocationResult invocation;
    invocation.index = invocation_index + 1;
    if (invocation_index % 2 == 0) {
      invocation.order = "ordinary_then_partitioned";
      invocation.ordinary = run_ordinary(invocation.index);
      if (report.outcome != "measured") {
        report.invocations.clear();
        break;
      }
      invocation.partitioned = run_partitioned(invocation.index);
    } else {
      invocation.order = "partitioned_then_ordinary";
      invocation.partitioned = run_partitioned(invocation.index);
      if (report.outcome != "measured") {
        report.invocations.clear();
        break;
      }
      invocation.ordinary = run_ordinary(invocation.index);
    }

    if (report.outcome != "measured") {
      report.invocations.clear();
      break;
    }
    invocation.ordinary_median_ms =
        median(invocation.ordinary.gpu_latency_ms);
    invocation.partitioned_median_ms =
        median(invocation.partitioned.gpu_latency_ms);
    if (!(invocation.ordinary_median_ms > 0.0) ||
        !(invocation.partitioned_median_ms > 0.0) ||
        !std::isfinite(invocation.ordinary_median_ms) ||
        !std::isfinite(invocation.partitioned_median_ms)) {
      throw BenchError("measured latency is non-positive or non-finite");
    }
    invocation.improvement_percent =
        100.0 * (invocation.ordinary_median_ms -
                 invocation.partitioned_median_ms) /
        invocation.ordinary_median_ms;
    invocation.gate_pass = invocation.improvement_percent >= 10.0 &&
                           invocation.ordinary.correctness &&
                           invocation.partitioned.correctness;
    report.invocations.push_back(std::move(invocation));
  }

  buffers.release();
  BKL_CUDA_STAGE("synchronize device before reporting", cudaDeviceSynchronize());

  if (report.outcome == "measured") {
    if (report.invocations.size() != kIndependentInvocations) {
      throw BenchError("measurement did not produce three invocations");
    }
    report.follow_up_justified =
        std::all_of(report.invocations.begin(), report.invocations.end(),
                    [](const InvocationResult& invocation) {
                      return invocation.gate_pass;
                    });
    std::vector<double> ordinary_invocation_medians;
    std::vector<double> partitioned_invocation_medians;
    ordinary_invocation_medians.reserve(report.invocations.size());
    partitioned_invocation_medians.reserve(report.invocations.size());
    for (const InvocationResult& invocation : report.invocations) {
      ordinary_invocation_medians.push_back(invocation.ordinary_median_ms);
      partitioned_invocation_medians.push_back(
          invocation.partitioned_median_ms);
    }
    report.ordinary_overall_median_ms = median(ordinary_invocation_medians);
    report.partitioned_overall_median_ms =
        median(partitioned_invocation_medians);
    report.overall_improvement_percent =
        100.0 * (report.ordinary_overall_median_ms -
                 report.partitioned_overall_median_ms) /
        report.ordinary_overall_median_ms;
  }

  write_report_atomically(output_path, report);
  print_summary(report, output_path);
  return 0;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path output_path;
  if (argc == 3 && std::string(argv[1]) == "--out") {
    output_path = argv[2];
  } else if (argc != 1) {
    std::fprintf(stderr, "usage: %s [--out PATH]\n", argv[0]);
    return 2;
  }

  try {
    return run_benchmark(output_path);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "bkl green_ctx_bench failed: %s\n", error.what());
    return 1;
  }
}
