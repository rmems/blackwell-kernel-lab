add_executable(bkl_device_hello device_hello.cu)
set_target_properties(bkl_device_hello PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src"
)
