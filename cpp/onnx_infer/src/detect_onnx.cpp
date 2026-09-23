// puck/knob 탐지 - ONNX Runtime C++ 추론
//
// YOLO26은 NMS-free 구조라서, 정적 shape(export_onnx.sh 참고)로 export하면
// 출력 텐서가 고정 [1, 300, 6] = (x1, y1, x2, y2, score, class_id) 형태로
// 나옵니다(공식 문서 기준). 별도 NMS 없이 score threshold + 좌표 rescale만
// 하면 됩니다.
//
// 주의: 이 코드는 "검증된 스펙" 기반 1차 구현입니다. 실제 export된 모델로
// 처음 실행할 때 아래처럼 output_shape을 꼭 출력해서 [1, 300, 6]이 맞는지,
// 좌표가 픽셀 단위(0~640)인지 정규화(0~1)인지 확인하고 필요하면 보정하세요.
//
// 빌드: ../CMakeLists.txt / ../README.md 참고

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

// data.yaml과 반드시 동일하게 유지: 0=puck, 1=knob
const std::vector<std::string> kClassNames = {"puck", "knob"};
constexpr int kImgSize = 640;

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

// 정적 이미지 파일 확장자인지 확인 (그 외는 전부 VideoCapture로 열어봄:
// 숫자 인덱스, /dev/realsense_color 같은 장치 경로, 동영상 파일, rtsp 등)
bool HasImageExtension(const std::string& path) {
    static const std::vector<std::string> kImageExts = {".jpg", ".jpeg", ".png", ".bmp"};
    for (const auto& ext : kImageExts) {
        if (path.size() >= ext.size() &&
            std::equal(ext.rbegin(), ext.rend(), path.rbegin(), [](char a, char b) {
                return std::tolower(static_cast<unsigned char>(a)) ==
                       std::tolower(static_cast<unsigned char>(b));
            })) {
            return true;
        }
    }
    return false;
}

// 종횡비 유지 리사이즈 + 패딩 (YOLO 표준 전처리)
cv::Mat Letterbox(const cv::Mat& src, int target, LetterboxInfo& info) {
    const float scale = std::min(static_cast<float>(target) / src.cols,
                                  static_cast<float>(target) / src.rows);
    const int new_w = static_cast<int>(std::round(src.cols * scale));
    const int new_h = static_cast<int>(std::round(src.rows * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    cv::Mat out(target, target, src.type(), cv::Scalar(114, 114, 114));
    info.pad_x = (target - new_w) / 2;
    info.pad_y = (target - new_h) / 2;
    resized.copyTo(out(cv::Rect(info.pad_x, info.pad_y, new_w, new_h)));
    info.scale = scale;
    return out;
}

std::vector<float> Preprocess(const cv::Mat& frame, LetterboxInfo& lb_info) {
    cv::Mat letterboxed = Letterbox(frame, kImgSize, lb_info);

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> input(1 * 3 * kImgSize * kImgSize);
    const size_t plane = static_cast<size_t>(kImgSize) * kImgSize;
    for (int c = 0; c < 3; ++c) {
        cv::Mat& ch = channels[c];
        if (!ch.isContinuous()) ch = ch.clone();
        std::memcpy(input.data() + c * plane, ch.ptr<float>(), plane * sizeof(float));
    }
    return input;
}

std::vector<Detection> Postprocess(const float* output, int64_t num_detections,
                                    int64_t num_values, const LetterboxInfo& lb_info,
                                    float score_thresh) {
    std::vector<Detection> detections;
    for (int64_t i = 0; i < num_detections; ++i) {
        const float* row = output + i * num_values;
        const float score = row[4];
        if (score < score_thresh) continue;

        Detection det;
        det.x1 = (row[0] - lb_info.pad_x) / lb_info.scale;
        det.y1 = (row[1] - lb_info.pad_y) / lb_info.scale;
        det.x2 = (row[2] - lb_info.pad_x) / lb_info.scale;
        det.y2 = (row[3] - lb_info.pad_y) / lb_info.scale;
        det.score = score;
        det.class_id = static_cast<int>(row[5]);
        detections.push_back(det);
    }
    return detections;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용법: " << argv[0]
                  << " <model.onnx> <image_path|camera_index|device_path> [score_thresh]\n"
                  << "  예: " << argv[0] << " best.onnx /dev/realsense_color 0.5\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string source = argv[2];
    const float score_thresh = argc > 3 ? std::stof(argv[3]) : 0.5f;

    // ---- ONNX Runtime 세션 초기화 ----
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "puck_knob_detect");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> input_names{input_name_alloc.get()};
    std::vector<const char*> output_names{output_name_alloc.get()};

    std::cout << "모델 로드 완료: " << model_path << "\n";
    std::cout << "  input: " << input_names[0] << " / output: " << output_names[0] << "\n";

    // ---- 입력 소스 판별 ----
    // .jpg/.jpeg/.png/.bmp 확장자만 정지 이미지로 취급.
    // 그 외(숫자 인덱스, /dev/realsense_color 같은 V4L2 장치 경로, .mp4 등)는
    // 전부 cv::VideoCapture로 엶 — OpenCV V4L2 백엔드는 장치 경로 문자열을
    // 그대로 받아들이므로 udev symlink(/dev/realsense_color 등)를 직접 넘기면 됨.
    const bool is_numeric_index =
        !source.empty() && std::all_of(source.begin(), source.end(), ::isdigit);
    const bool is_static_image = !is_numeric_index && HasImageExtension(source);
    const bool is_camera = !is_static_image;

    cv::VideoCapture cap;
    cv::Mat frame;
    if (is_camera) {
        if (is_numeric_index) {
            cap.open(std::stoi(source));
        } else {
            cap.open(source, cv::CAP_V4L2);
        }
        if (!cap.isOpened()) {
            std::cerr << "카메라/스트림을 열 수 없습니다: " << source << "\n";
            return 1;
        }
    } else {
        frame = cv::imread(source);
        if (frame.empty()) {
            std::cerr << "이미지를 읽을 수 없습니다: " << source << "\n";
            return 1;
        }
    }

    bool first_frame = true;
    do {
        if (is_camera) {
            cap >> frame;
            if (frame.empty()) break;
        }

        LetterboxInfo lb_info;
        std::vector<float> input_values = Preprocess(frame, lb_info);

        std::array<int64_t, 4> input_shape{1, 3, kImgSize, kImgSize};
        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_values.data(), input_values.size(), input_shape.data(),
            input_shape.size());

        auto output_tensors =
            session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                        output_names.data(), 1);

        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        if (first_frame) {
            std::cout << "출력 shape: [";
            for (size_t i = 0; i < output_shape.size(); ++i) {
                std::cout << output_shape[i] << (i + 1 < output_shape.size() ? ", " : "");
            }
            std::cout << "]  (예상: [1, 300, 6] - 다르면 코드 보정 필요)\n";
            first_frame = false;
        }

        const float* output_data = output_tensors[0].GetTensorMutableData<float>();
        const int64_t num_detections = output_shape[1];
        const int64_t num_values = output_shape[2];

        auto detections =
            Postprocess(output_data, num_detections, num_values, lb_info, score_thresh);

        for (const auto& det : detections) {
            const std::string label = det.class_id >= 0 &&
                                               det.class_id < static_cast<int>(kClassNames.size())
                                           ? kClassNames[det.class_id]
                                           : "unknown(" + std::to_string(det.class_id) + ")";
            std::cout << label << " " << det.score << " [" << det.x1 << ", " << det.y1 << ", "
                      << det.x2 << ", " << det.y2 << "]\n";

            cv::rectangle(frame, cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                          cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)),
                          cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, label, cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1) - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        if (is_camera) {
            cv::imshow("puck_knob_detect", frame);
            if (cv::waitKey(1) == 27) break;  // ESC로 종료
        } else {
            cv::imwrite("detect_result.jpg", frame);
            std::cout << "결과 저장: detect_result.jpg\n";
        }
    } while (is_camera);

    return 0;
}
