# First-party L3 kernel workspace for blackwell-kernel-lab (sm_120).
# CPU CI: configure with -DBKL_ENABLE_CUDA=OFF (no toolkit required).
cmake_minimum_required(VERSION 3.24)

project(blackwell_kernels LANGUAGES CXX)

option(BKL_ENABLE_CUDA "Build CUDA kernels (requires nvcc + toolkit)" ON)
# CUDA 13.3 host-compiler cap: system gcc 16 is too new; prefer gcc-15 if present.
set(BKL_CUDA_HOST_COMPILER "" CACHE FILEPATH "Host C++ compiler for nvcc")

if(NOT BKL_ENABLE_CUDA)
  message(STATUS "BKL_ENABLE_CUDA=OFF — skipping kernel targets (CPU CI path)")
  return()
endif()

if(NOT CMAKE_CUDA_COMPILER)
  if(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "nvcc" FORCE)
  endif()
endif()

# Prefer a host compiler CUDA 13.3 accepts (gcc ≤ 15).
if(BKL_CUDA_HOST_COMPILER STREQUAL "" OR BKL_CUDA_HOST_COMPILER STREQUAL "BKL_CUDA_HOST_COMPILER-NOTFOUND")
  foreach(_cc
      "/home/linuxbrew/.linuxbrew/bin/g++-15"
      "/usr/bin/g++-15"
      "/usr/bin/g++-14"
      "/usr/bin/g++-13")
    if(EXISTS "${_cc}")
      set(BKL_CUDA_HOST_COMPILER "${_cc}" CACHE FILEPATH "Host C++ compiler for nvcc" FORCE)
      break()
    endif()
  endforeach()
endif()

if(BKL_CUDA_HOST_COMPILER AND EXISTS "${BKL_CUDA_HOST_COMPILER}")
  set(CMAKE_CUDA_HOST_COMPILER "${BKL_CUDA_HOST_COMPILER}" CACHE FILEPATH "" FORCE)
  # Also help nvcc via flags if CMake still probes with system gcc.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -ccbin=${BKL_CUDA_HOST_COMPILER}")
  message(STATUS "CUDA host compiler: ${BKL_CUDA_HOST_COMPILER}")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  message(STATUS "CUDA host compiler: system default (+ -allow-unsupported-compiler)")
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(WARNING
    "CUDA toolkit / nvcc not found; skipping kernel build. "
    "Install CUDA 13.3+ for sm_120, or pass -DBKL_ENABLE_CUDA=OFF to silence.")
  return()
endif()

enable_language(CUDA)
# RTX 5080 consumer Blackwell — compute capability 12.0
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA arch for this lab" FORCE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)
