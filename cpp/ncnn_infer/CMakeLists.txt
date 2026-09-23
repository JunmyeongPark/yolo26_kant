cmake_minimum_required(VERSION 3.14)
project(puck_knob_detect_ncnn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NCNN 위치 지정:
#   cmake -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn ..
find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)  # 전처리/시각화용

add_executable(detect_ncnn src/detect_ncnn.cpp)

target_link_libraries(detect_ncnn PRIVATE
    ncnn
    ${OpenCV_LIBS}
)
