cmake_minimum_required (VERSION 2.6)
project (codecsimulation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========== 线程库 ==========
find_package(Threads REQUIRED)

# ========== libyuv ==========
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libyuv)

# ========== x265 ==========
set(ENABLE_SHARED OFF CACHE BOOL "Build shared libx265" FORCE)
set(ENABLE_CLI    OFF CACHE BOOL "Build x265 CLI"        FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/x265/source
                 ${CMAKE_CURRENT_BINARY_DIR}/x265_build)

# ========== ExternalProject (x264 / vvenc) ==========
include(ExternalProject)

# ---- x264 ----
ExternalProject_Add(x264_proj
    SOURCE_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/x264
    CONFIGURE_COMMAND ./configure --enable-pic --enable-static --disable-cli --prefix=<INSTALL_DIR>
    BUILD_COMMAND     $(MAKE)
    INSTALL_COMMAND   $(MAKE) install
    BUILD_IN_SOURCE   1
)

ExternalProject_Get_Property(x264_proj install_dir)
set(x264_install_dir ${install_dir})

file(MAKE_DIRECTORY ${x264_install_dir}/include)

add_library(x264 STATIC IMPORTED GLOBAL)
set_target_properties(x264 PROPERTIES
    IMPORTED_LOCATION             ${x264_install_dir}/lib/libx264.a
    INTERFACE_INCLUDE_DIRECTORIES ${x264_install_dir}/include
)
add_dependencies(x264 x264_proj)

# ---- vvenc ----
ExternalProject_Add(vvenc_proj
    SOURCE_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/vvenc
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DVVENC_ENABLE_X86_SIMD=ON
        -DVVENC_ENABLE_WERROR=OFF
    BUILD_COMMAND     $(MAKE)
    INSTALL_COMMAND   $(MAKE) install
)

ExternalProject_Get_Property(vvenc_proj install_dir)
set(vvenc_install_dir ${install_dir})

file(MAKE_DIRECTORY ${vvenc_install_dir}/include)

add_library(vvenc STATIC IMPORTED GLOBAL)
set_target_properties(vvenc PROPERTIES
    IMPORTED_LOCATION             ${vvenc_install_dir}/lib/libvvenc.a   # 若实际是 .so 改这里
    INTERFACE_INCLUDE_DIRECTORIES ${vvenc_install_dir}/include
)
add_dependencies(vvenc vvenc_proj)

# ========== 可执行程序 ==========
add_executable(testcodec
    x264Codec.cpp
    x265Codec.cpp
    x266Codec.cpp
    test.cpp
)

target_include_directories(testcodec PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/libyuv/include
    ${CMAKE_CURRENT_SOURCE_DIR}/x265/source
)

target_link_libraries(testcodec PRIVATE
    x264
    x265
    vvenc
    yuv
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# ========== pybind11 模块 ==========
add_subdirectory(pybind11)

pybind11_add_module(codecsimulator 
    x264Codec.cpp
    x265Codec.cpp
    x266Codec.cpp
    binding.cpp
)

target_include_directories(codecsimulator PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/libyuv/include
    ${CMAKE_CURRENT_SOURCE_DIR}/x265/source
)

target_link_libraries(codecsimulator PRIVATE
    x264
    x265
    vvenc
    yuv
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
