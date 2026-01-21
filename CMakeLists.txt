cmake_minimum_required(VERSION 3.15...3.27)
project(bspot-python)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

set(CMAKE_CXX_STANDARD 20)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (MSVC)
  # MSVC's default OpenMP runtime lacks task support; use the LLVM runtime.
  set(OpenMP_RUNTIME_MSVC "llvm")
endif()

find_package(OpenMP REQUIRED)

add_subdirectory(ext/nanobind)
add_subdirectory(ext/eigen)

# Default double precision version
nanobind_add_module(pybspot module.cpp)
target_link_libraries(pybspot PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_include_directories(pybspot PRIVATE ${Python_INCLUDE_DIRS})
install(TARGETS pybspot LIBRARY DESTINATION . RUNTIME DESTINATION .)

# Single precision version
nanobind_add_module(pybspot_f32 module.cpp)
target_compile_definitions(pybspot_f32 PRIVATE BSPOT_SINGLE_PRECISION)
target_link_libraries(pybspot_f32 PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_include_directories(pybspot_f32 PRIVATE ${Python_INCLUDE_DIRS})
install(TARGETS pybspot_f32 LIBRARY DESTINATION . RUNTIME DESTINATION .)
