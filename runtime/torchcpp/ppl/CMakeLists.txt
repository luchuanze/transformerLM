cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(speakers VERSION 0.1)

option(GRPC "whether to build gRPC" OFF)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/kaldi  # for not changing c++ header names in kaldi source files
)
set(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
# get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
# set(FETCHCONTENT_BASE_DIR ${fc_base})
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.1")
if(NOT MSVC)
  # Keep the same with openfst, -fPIC or -fpic
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
endif()


# third_party: libtorch use FetchContent_Declare to download, and
# use find_package to find since libtorch is not a standard cmake project
set(PYTORCH_VERSION "1.13.0")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
    set(URL_HASH "SHA256=d7043b7d7bdb5463e5027c896ac21b83257c32c533427d4d0d7b251548db8f4b")
    set(CMAKE_BUILD_TYPE "Release")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
    set(URL_HASH "SHA256=6b99edc046f37ad37a3b04dc86663895f10c362af09fdd10884f30360d0ba023")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${PYTORCH_VERSION}.zip")
    set(URL_HASH "SHA256=07cac2c36c34f13065cb9559ad5270109ecbb468252fb0aeccfd89322322a2b5")
else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux' or 'Darwin')")
endif()
FetchContent_Declare(libtorch
  URL      ${LIBTORCH_URL}
)
set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories("build/_deps/libtorch-src/include") 
include_directories("build/_deps/libtorch-src/include/torch/csrc/api/include") 
include_directories("src")


add_library(lmppl SHARED
 src/torch_lm_model.cpp
 src/torch_lm_inference.cpp
 src/lm_ppl.cpp
)

target_link_libraries(lmppl PUBLIC ${TORCH_LIBRARIES})

add_executable(ppl_test bin/ppl_test.cpp)
target_link_libraries(ppl_test PUBLIC lmppl)