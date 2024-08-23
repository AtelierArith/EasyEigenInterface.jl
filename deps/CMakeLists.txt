cmake_minimum_required(VERSION 3.22)
project(main)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

find_package(Eigen3 REQUIRED)
find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
message(STATUS "Found JlCxx at ${JlCxx_location}")
message(STATUS "Julia_LIBRARY at ${Julia_LIBRARY}")

add_library(jl_easy_eigen_interface SHARED jl_easy_eigen_interface.cpp)
target_link_libraries(jl_easy_eigen_interface JlCxx::cxxwrap_julia ${Julia_LIBRARY} Eigen3::Eigen)
