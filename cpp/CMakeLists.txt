cmake_minimum_required (VERSION 2.6)
project (codecsimulation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libyuv)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vvenc)
set(ENABLE_SHARED OFF CACHE BOOL "Build shared libx265" FORCE)
set(ENABLE_CLI OFF CACHE BOOL "Build x265 CLI" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/x265/source ${CMAKE_CURRENT_BINARY_DIR}/x265_build)

include(ExternalProject)
ExternalProject_Add(x264_proj
    SOURCE_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/x264
    CONFIGURE_COMMAND ./configure --enable-pic --enable-static --disable-cli --prefix <INSTALL_DIR>
    BUILD_COMMAND     $(MAKE)
    INSTALL_COMMAND   $(MAKE) install
    BUILD_IN_SOURCE   1
)
ExternalProject_Get_Property(x264_proj install_dir)
add_library(x264 STATIC IMPORTED GLOBAL)
set_target_properties(x264 PROPERTIES IMPORTED_LOCATION ${install_dir}/lib/libx264.a)
add_dependencies(x264 x264_proj)

add_executable(testcodec 
               x264Codec.cpp
               x265Codec.cpp
               x266Codec.cpp
               test.cpp)
target_include_directories(testcodec PUBLIC 
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}/libyuv/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/vvenc/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/x264
                           ${CMAKE_CURRENT_SOURCE_DIR}/x265/source
)
target_link_libraries(testcodec
                      x264
                      x265
                      vvenc
                      yuv)

add_subdirectory(pybind11)
pybind11_add_module(codecsimulator 
                    x264Codec.cpp
                    x265Codec.cpp
                    x266Codec.cpp
                    binding.cpp)
target_include_directories(codecsimulator PUBLIC 
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}/libyuv/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/vvenc/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/x264
                           ${CMAKE_CURRENT_SOURCE_DIR}/x265/source
                           )
target_link_libraries(codecsimulator
                      x264
                      x265
                      vvenc
                      yuv)
