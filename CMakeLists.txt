#
# wavelet_scatter
cmake_minimum_required (VERSION 3.0)
project(wavelet_scatter VERSION 0.1.0)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR /home/seeking/Downloads/opencv-3.4.2/install/share/OpenCV)

SET(PLATFORM_COMPILE_FLAGS "-march=native")
SET(COMMON_COMPILE_FLAGS "-fopenmp ")
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package (OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(Armadillo)
include_directories (${OpenCV_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)


add_executable(test_wst test_wst.cpp src/wst.cpp)
target_link_libraries (test_wst  ${OpenCV_LIBS} ${ARMADILLO_LIB_DIRS} -lpthread -fopenmp)
