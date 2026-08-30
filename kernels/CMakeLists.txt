# First-party L3 kernel workspace for blackwell-kernel-lab (sm_120).
# CPU CI: configure with -DBKL_ENABLE_CUDA=OFF (no toolkit required).
cmake_minimum_required(VERSION 3.24)

project(blackwell_kernels LANGUAGES CXX)

option(BKL_ENABLE_CUDA "Build CUDA kernels (requires nvcc + toolkit)" ON)
# CUDA 13.3 supports host GCC 6.x–15.x only. Prefer g++-15 when present.
option(BKL_ALLOW_UNSUPPORTED_HOST_COMPILER
  "Opt in to -allow-unsupported-compiler (e.g. system gcc 16). Default OFF."
  OFF)
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

# Host compiler precedence (CUDA 13.3: GCC 6.x–15.x):
#   1) CMAKE_CUDA_HOST_COMPILER already set (e.g. -D… or prior cache)
#   2) CUDAHOSTCXX env
#   3) -DBKL_CUDA_HOST_COMPILER=…
#   4) auto-discover g++-15 … g++-6
#   5) BKL_ALLOW_UNSUPPORTED_HOST_COMPILER opt-in, else FATAL
#
# Do not FORCE-overwrite an explicit host compiler. CMAKE_CUDA_HOST_COMPILER is
# enough for nvcc — do not also inject -ccbin into CMAKE_CUDA_FLAGS.

if(CMAKE_CUDA_HOST_COMPILER AND EXISTS "${CMAKE_CUDA_HOST_COMPILER}")
  message(STATUS "CUDA host compiler (CMAKE_CUDA_HOST_COMPILER): ${CMAKE_CUDA_HOST_COMPILER}")
elseif(DEFINED ENV{CUDAHOSTCXX} AND EXISTS "$ENV{CUDAHOSTCXX}")
  set(CMAKE_CUDA_HOST_COMPILER "$ENV{CUDAHOSTCXX}" CACHE FILEPATH "Host C++ compiler for nvcc")
  message(STATUS "CUDA host compiler (CUDAHOSTCXX): ${CMAKE_CUDA_HOST_COMPILER}")
else()
  if(BKL_CUDA_HOST_COMPILER STREQUAL "" OR BKL_CUDA_HOST_COMPILER STREQUAL "BKL_CUDA_HOST_COMPILER-NOTFOUND")
    # Prefer newer supported compilers first (15 … 6).
    foreach(_cc
        "/home/linuxbrew/.linuxbrew/bin/g++-15"
        "/usr/bin/g++-15"
        "/usr/bin/g++-14"
        "/usr/bin/g++-13"
        "/usr/bin/g++-12"
        "/usr/bin/g++-11"
        "/usr/bin/g++-10"
        "/usr/bin/g++-9"
        "/usr/bin/g++-8"
        "/usr/bin/g++-7"
        "/usr/bin/g++-6")
      if(EXISTS "${_cc}")
        set(BKL_CUDA_HOST_COMPILER "${_cc}" CACHE FILEPATH "Host C++ compiler for nvcc")
        break()
      endif()
    endforeach()
  endif()

  if(BKL_CUDA_HOST_COMPILER AND EXISTS "${BKL_CUDA_HOST_COMPILER}")
    set(CMAKE_CUDA_HOST_COMPILER "${BKL_CUDA_HOST_COMPILER}" CACHE FILEPATH "Host C++ compiler for nvcc")
    message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
  elseif(BKL_ALLOW_UNSUPPORTED_HOST_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    message(WARNING
      "BKL_ALLOW_UNSUPPORTED_HOST_COMPILER=ON — using system host compiler "
      "with -allow-unsupported-compiler (not official for CUDA 13.3 + gcc 16).")
  else()
    message(FATAL_ERROR
      "No supported CUDA host compiler (g++ 6–15) found.\n"
      "  Install: brew install gcc@15\n"
      "  Or set:  -DBKL_CUDA_HOST_COMPILER=/path/to/g++-15\n"
      "  Or set:  -DCMAKE_CUDA_HOST_COMPILER=… / CUDAHOSTCXX=…\n"
      "  Or skip: -DBKL_ENABLE_CUDA=OFF\n"
      "  Opt-in unsupported: -DBKL_ALLOW_UNSUPPORTED_HOST_COMPILER=ON")
  endif()
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
find_package(CUDAToolkit REQUIRED)
# RTX 5080 consumer Blackwell — compute capability 12.0
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA arch for this lab" FORCE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)
