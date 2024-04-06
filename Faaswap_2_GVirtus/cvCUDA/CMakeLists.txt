cmake_minimum_required(VERSION 3.10)
project(CVModel)

# Set the build type if it is not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Add executable
add_executable(cvModel cvModel.cpp)

link_directories(
        /root/zzx/GVirtuSPlus/lib
        /root/zzx/GVirtuSPlus/lib/frontend
)

include_directories(/usr/local/cuda-11.4/targets/x86_64-linux/include)

# Specify libraries to link with the target
target_link_libraries(cvModel
        /root/zzx/GVirtuSPlus/lib/frontend/libcudart.so
        /root/zzx/GVirtuSPlus/lib/frontend/libcudnn.so
        /root/zzx/GVirtuSPlus/lib/frontend/libcufft.so
        /root/zzx/GVirtuSPlus/lib/frontend/libcublas.so
        /root/zzx/GVirtuSPlus/lib/libgvirtus-communicators-tcp.so
)

# Set RPATH
set_target_properties(cvModel PROPERTIES
        INSTALL_RPATH "/root/zzx/GVirtuSPlus/lib;/root/zzx/GVirtuSPlus/lib/frontend"
        BUILD_WITH_INSTALL_RPATH TRUE
)
