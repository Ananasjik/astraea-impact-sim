cmake_minimum_required(VERSION 3.24)
project(AstraeaCollision LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(sim_engine STATIC
    engine/eos.cu
    engine/sph_gravity.cu
    engine/presets.cpp
    engine/planet_builder.cpp
    engine/simulation.cpp
)

target_include_directories(sim_engine PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/engine"
)

target_compile_options(sim_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(sim_engine PUBLIC
    CUDA::cudart
    Threads::Threads
)

add_executable(sim_headless
    engine/headless_main.cpp
)
target_link_libraries(sim_headless PRIVATE sim_engine)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/server/CMakeLists.txt")
    add_subdirectory(server)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
