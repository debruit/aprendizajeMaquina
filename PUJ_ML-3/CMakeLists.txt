## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

## -- Base configuration
cmake_minimum_required(VERSION 3.18)
project(PUJ_ML VERSION 1.0.0.0)

## -- Some basic configurations
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## -- CMake directives
include(GenerateExportHeader)

## -- Find required packages
find_package(Boost REQUIRED program_options)
find_package(Eigen3 REQUIRED)

## -- Source code
subdirs(lib/cxx examples/cxx)

## eof - $RCSfile$
