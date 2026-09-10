add_executable(bkl_device_hello device_hello.cu)

add_executable(bkl_graph_launch_bench graph_launch_bench.cu)

add_executable(bkl_green_ctx_bench green_ctx_bench.cu)
target_link_libraries(bkl_green_ctx_bench PRIVATE CUDA::cuda_driver)

# README, docs/CI.md, the recipes, and ci-gpu.yml all invoke these binaries as
# build/kernels/src/<target>. Pin that for every target instead of letting two
# of the three inherit it incidentally from CMAKE_CURRENT_BINARY_DIR.
set_target_properties(
  bkl_device_hello
  bkl_graph_launch_bench
  bkl_green_ctx_bench
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src"
)
