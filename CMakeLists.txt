SET(GCC_COVERAGE_COMPILE_FLAGS "-O3 -fopenmp -Wall -pedantic")
SET(GCC_COVERAGE_LINK_FLAGS    "-fopenmp")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )
cmake_minimum_required(VERSION 2.8)
project( motld )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( camexample camExample.cpp )
target_link_libraries( camexample ${OpenCV_LIBS} )
