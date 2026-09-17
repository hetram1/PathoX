cmake_minimum_required(VERSION 3.18)
project(pathox_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(pathox_native bindings.cpp)

target_compile_features(pathox_native PRIVATE cxx_std_17)
target_link_libraries(pathox_native PRIVATE OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pathox_native PRIVATE -O3 -march=native)
endif()
