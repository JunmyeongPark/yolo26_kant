cmake_minimum_required(VERSION 3.14)
project(puck_knob_detect_onnx CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# ONNX Runtime 위치 지정:
#   cmake -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime-linux-... ..
if(NOT DEFINED ONNXRUNTIME_ROOTDIR)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOTDIR 이 설정되지 않았습니다. "
        "cmake -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime-linux-... .. 로 지정하세요. "
        "(README.md 참고)")
endif()

add_executable(detect_onnx src/detect_onnx.cpp)

target_include_directories(detect_onnx PRIVATE
    ${ONNXRUNTIME_ROOTDIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(detect_onnx PRIVATE
    ${ONNXRUNTIME_ROOTDIR}/lib
)

target_link_libraries(detect_onnx PRIVATE
    onnxruntime
    ${OpenCV_LIBS}
)

# 실행 시 libonnxruntime.so를 못 찾는 문제 방지 (rpath에 심어둠)
set_target_properties(detect_onnx PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_ROOTDIR}/lib
    INSTALL_RPATH ${ONNXRUNTIME_ROOTDIR}/lib
)
