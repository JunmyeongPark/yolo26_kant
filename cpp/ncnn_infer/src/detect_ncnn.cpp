// puck/knob 탐지 - NCNN C++ 추론 (스텁 / 아직 미완성)
//
// ================================ 아직 완성되지 않음 ================================
// NCNN이 라즈베리파이에서 가장 빠른 추론 성능을 내지만(Ultralytics 공식 벤치마크
// 기준 ONNX/MNN보다 빠름), YOLO26 + NCNN 조합의 C++ 출력 텐서 스펙(블롭 이름,
// shape)을 아직 검증된 문서로 확인하지 못해서 파싱 로직을 비워뒀습니다.
//
// 채우는 방법:
//   1. scripts/export_ncnn.sh 로 실제 모델을 NCNN으로 export
//   2. 생성된 <모델명>_ncnn_model/model.ncnn.param 파일을 텍스트 에디터로 열어서
//      - input 블롭 이름 (보통 "in0" 류)
//      - output 블롭 이름 (보통 "out0" 류)
//      - Extract 아래쪽에 명시된 output 개수/이름 확인
//   3. 아래 main()에서 net.load_param()/load_model() 경로와
//      ex.input("<input_blob>", in) / ex.extract("<output_blob>", out) 이름을
//      실제 값으로 교체
//   4. out.w, out.h, out.c 를 출력해서 실제 shape을 확인하고, cpp/onnx_infer 쪽
//      [1, 300, 6] NMS-free 스펙과 같은 형태인지 대조 후 Postprocess 로직 완성
//      (아마 동일하게 x1,y1,x2,y2,score,class_id 형태일 가능성이 높지만 실측 필요)
//
// 이 파일이 완성되면 cpp/onnx_infer/src/detect_onnx.cpp 와 동일한 CLI
// (model_path, image/camera source, score_thresh)를 유지해서 서로 바꿔 끼울 수
// 있게 맞춰두는 걸 권장합니다.
// =====================================================================================

#include <net.h>  // ncnn

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용법: " << argv[0]
                  << " <ncnn_model_dir> <image_path|camera_index> [score_thresh]\n";
        return 1;
    }

    const std::string model_dir = argv[1];
    // const std::string source = argv[2];
    // const float score_thresh = argc > 3 ? std::stof(argv[3]) : 0.5f;

    ncnn::Net net;

    // TODO: 실제 export된 파일명으로 교체 (보통 model.ncnn.param / model.ncnn.bin)
    if (net.load_param((model_dir + "/model.ncnn.param").c_str()) != 0) {
        std::cerr << "param 로드 실패\n";
        return 1;
    }
    if (net.load_model((model_dir + "/model.ncnn.bin").c_str()) != 0) {
        std::cerr << "bin 로드 실패\n";
        return 1;
    }

    std::cout << "NCNN 모델 로드 완료: " << model_dir << "\n";
    std::cout << "TODO: 아래부터 전처리/추론/후처리 구현 필요 (파일 상단 주석 참고)\n";

    // TODO: 전처리 (letterbox + normalize) -> ncnn::Mat 변환
    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(...);
    // in.substract_mean_normalize(mean_vals, norm_vals);

    // TODO: 추론
    // ncnn::Extractor ex = net.create_extractor();
    // ex.input("in0", in);          // <- 실제 input 블롭 이름으로 교체
    // ncnn::Mat out;
    // ex.extract("out0", out);      // <- 실제 output 블롭 이름으로 교체

    // TODO: out.w / out.h / out.c 출력해서 실제 shape 확인 후 후처리 구현
    // std::cout << "output shape: w=" << out.w << " h=" << out.h << " c=" << out.c << "\n";

    return 0;
}
