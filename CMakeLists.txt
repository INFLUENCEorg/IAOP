cmake_minimum_required (VERSION 3.0)
project (MAPS)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

add_definitions(-DC10_USE_GLOG)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -w")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(third-party/yaml-cpp)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(src third-party/headers)

find_library(LIB_glog glog)

add_executable(main src/main.cpp)

find_package(Torch REQUIRED PATHS third-party/libtorch/)

target_link_libraries(main ${LIB_glog} yaml-cpp ${TORCH_LIBRARIES})
