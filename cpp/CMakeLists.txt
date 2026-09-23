cmake_minimum_required(VERSION 3.14)
project(puck_knob_cpp)

add_subdirectory(onnx_infer)

# NCNN 쪽은 detect_ncnn.cpp가 스텁 상태라 기본으로는 빌드에서 제외.
# 채워 넣은 뒤 아래 주석 해제:
# add_subdirectory(ncnn_infer)
