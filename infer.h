#ifndef INFER_H
#define INFER_H

#include "./lib/onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
#include <vector>
#include <string>
#include <memory>  // For std::unique_ptr
#include "lib/nlohmann/json.hpp"

using json = nlohmann::json;

namespace infer {

// Load scaler params from JSON
void load_scaler(const std::string& path,
                 std::vector<float>& mean,
                 std::vector<float>& scale);

// Load an ONNX model into a unique_ptr<Ort::Session>
std::unique_ptr<Ort::Session> loadModel(const std::string& model_path, Ort::Env& env);

// Returns predicted price in PHP
float predict(const std::vector<float>& input_vals,
              const std::vector<float>& scaler_mean,
              const std::vector<float>& scaler_scale,
              Ort::Session& session);

}  // namespace infer

#endif // INFER_H
