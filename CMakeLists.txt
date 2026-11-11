cmake_minimum_required(VERSION 3.20)
project(RAMP LANGUAGES CXX)

# Build outside the source directory:
# cmake -S . -B /path/to/build -DCMAKE_BUILD_TYPE=Release \
#   -DHNSWLIB_ROOT=/path/to/hnswlib -DRABITQ_ROOT=/path/to/RaBitQ-Library \
#   -DFAISS_ROOT=/path/to/faiss
# cmake --build /path/to/build -j
# Dependencies: C++20, OpenSSL, OpenMP, BLAS/LAPACK, hnswlib, Faiss (CPU),
# and RaBitQ-Library (quantize_full_single and packed extended-code kernels).
# Third-party sources and binaries are installed separately, not vendored here.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenSSL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_path(
    HNSWLIB_INCLUDE_DIR hnswlib/hnswlib.h
    HINTS ${HNSWLIB_ROOT}
    PATH_SUFFIXES include REQUIRED)
find_path(
    RABITQ_INCLUDE_DIR rabitqlib/quantization/rabitq.hpp
    HINTS ${RABITQ_ROOT}
    PATH_SUFFIXES include REQUIRED)
find_library(
    RABITQ_LIBRARY rabitq_core
    HINTS ${RABITQ_ROOT}
    PATH_SUFFIXES lib lib64 build REQUIRED)
find_path(
    FAISS_INCLUDE_DIR faiss/Clustering.h
    HINTS ${FAISS_ROOT}
    PATH_SUFFIXES include REQUIRED)
find_library(
    FAISS_LIBRARY faiss
    HINTS ${FAISS_ROOT}
    PATH_SUFFIXES lib lib64 build/faiss build REQUIRED)

add_executable(ramp src/main.cpp)
target_include_directories(ramp PRIVATE include)
target_include_directories(
    ramp SYSTEM PRIVATE ${HNSWLIB_INCLUDE_DIR} ${RABITQ_INCLUDE_DIR}
                        ${RABITQ_INCLUDE_DIR}/rabitqlib/third ${FAISS_INCLUDE_DIR})
target_link_libraries(ramp PRIVATE ${RABITQ_LIBRARY} ${FAISS_LIBRARY} ${LAPACK_LIBRARIES}
                                   ${BLAS_LIBRARIES} OpenSSL::Crypto OpenMP::OpenMP_CXX)
target_compile_options(ramp PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-deprecated-declarations>)
option(RAMP_NATIVE "Optimize for the build machine, as in the experiments" OFF)
if(RAMP_NATIVE)
    target_compile_options(ramp PRIVATE -march=native)
endif()
