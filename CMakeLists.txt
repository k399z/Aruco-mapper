# @author        Liang Huanyan
# @email         lianghuanyan@alientek.com
# http://www.openedv.com/forum.php

cmake_minimum_required(VERSION 3.8)
message(STATUS "cmake version ${CMAKE_VERSION}")

set(TOOLCHAIN_DIR /opt/atk-dlrv1126-toolchain)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/usr/bin/arm-linux-gnueabihf-g++)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/usr/bin/arm-linux-gnueabihf-gcc)
set(SYSROOT ${TOOLCHAIN_DIR}/arm-buildroot-linux-gnueabihf/sysroot/usr/include)
set(CMAKE_SYSROOT ${TOOLCHAIN_DIR}/arm-buildroot-linux-gnueabihf/sysroot)

# 让编译/链接都使用 sysroot
add_compile_options(--sysroot=${CMAKE_SYSROOT})
add_link_options(
  --sysroot=${CMAKE_SYSROOT}
  -Wl,-rpath-link,${CMAKE_SYSROOT}/lib
  -Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib
  -Wl,--dynamic-linker=/lib/ld-linux-armhf.so.3
  -Wl,-rpath-link,${OPENCV_SYSROOT}/lib
  -Wl,-rpath-link,${OPENCV_SYSROOT}/usr/lib
  -Wl,-rpath,/usr/lib
)

# 保险：把 sysroot 的 include/lib 也加上
include_directories(${CMAKE_SYSROOT}/usr/include)
link_directories(${CMAKE_SYSROOT}/usr/lib)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-g -O0 -ggdb -gdwarf -funwind-tables -rdynamic)
add_definitions(-Wno-write-strings -Wno-return-type)


# === OpenCV 指向 Buildroot 输出（优先 host/sysroot，找不到再回退到 target） ===
set(OPENSDK $ENV{HOME}/rv1126-rv1109-linux/buildroot/output/rockchip_rv1126_rv1109_facial_gate)
set(OPENCV_SYSROOT ${OPENSDK}/host/arm-buildroot-linux-gnueabihf/sysroot)
if(NOT EXISTS "${OPENCV_SYSROOT}/usr/include/opencv2/aruco.hpp")
  message(WARNING "opencv2/aruco.hpp 未在 host/sysroot 找到，回退使用 target")
  set(OPENCV_SYSROOT ${OPENSDK}/target)
endif()

# 仅对 OpenCV 使用新的 include/lib 路径（其它保持你原来的 SYSROOT 不变）
include_directories(BEFORE ${OPENCV_SYSROOT}/usr/include)
link_directories(${OPENCV_SYSROOT}/usr/lib)


set(OPENCV_LIBS opencv_core opencv_imgcodecs opencv_imgproc opencv_features2d opencv_flann opencv_highgui opencv_freetype     opencv_videoio opencv_calib3d opencv_aruco)
set(RKNN_LIBS easymedia rga rkaiq rknn_api)
link_directories(include lib)
link_directories(${PROJECT_SOURCE_DIR}/librtsp)
	
include_directories(${SYSROOT})
include_directories(${SYSROOT}/rga)
include_directories(${SYSROOT}/easymedia)
include_directories(${SYSROOT}/rkaiq/uAPI)
include_directories(${SYSROOT}/rkaiq/xcore)
include_directories(${SYSROOT}/rkaiq/algos)
include_directories(${SYSROOT}/rkaiq/common)
include_directories(${SYSROOT}/rkaiq/iq_parser)
include_directories(${SYSROOT}/rknn)
include_directories(.)
include_directories(./include)
include_directories(./include/3rdparty)
include_directories(./include/3rdparty/drm/include/libdrm)
include_directories(./include/3rdparty/rga/include)

add_definitions(-DRKAIQ)

add_executable(main atk_yolov5_object_recognize.cpp )
target_link_libraries(main ${OPENCV_LIBS} ${RKNN_LIBS} rtsp pthread dl     avformat
avcodec
avutil
swscale
swresample
drm)