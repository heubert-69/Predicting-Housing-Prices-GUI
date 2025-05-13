#include <iostream>
#include "infer.h"
#include <fstream>
#include <stdexcept>

namespace infer {

void load_scaler(const std::string& path,
                 std::vector<float>& mean,
                 std::vector<float>& scale) {
    std::ifstream in(path);
    if (!in.is_open()) throw std::runtime_error("Failed to open scaler file: " + path);
    json js;
    in >> js;
    mean  = js["mean"].get<std::vector<float>>();
    scale = js["scale"].get<std::vector<float>>();
}

std::unique_ptr<Ort::Session> loadModel(const std::string& model_path, Ort::Env& env) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    return std::make_unique<Ort::Session>(env, model_path.c_str(), opts);
}

float predict(const std::vector<float>& input_vals,
              const std::vector<float>& scaler_mean,
              const std::vector<float>& scaler_scale,
              Ort::Session& session) {
    if (input_vals.size() != scaler_mean.size()) {
        throw std::invalid_argument("Input size does not match scaler parameters");
    }

    // Scale the input
    std::vector<float> scaled(input_vals.size());
    for (size_t i = 0; i < input_vals.size(); ++i) {
        scaled[i] = (input_vals[i] - scaler_mean[i]) / scaler_scale[i];
    }

    // Prepare tensor
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Shape of the tensor, assuming 2D input with 1 row and N columns
    std::vector<int64_t> shape = {1, static_cast<int64_t>(scaled.size())};

    // Create input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, scaled.data(), scaled.size(), shape.data(), shape.size());

    // Get input and output names
    auto input_name_strs = session.GetInputNames();
    auto output_name_strs = session.GetOutputNames();

    // Convert names from strings to const char*
    std::vector<const char*> input_names;
    for (const auto& s : input_name_strs) input_names.push_back(s.c_str());

    std::vector<const char*> output_names;
    for (const auto& s : output_name_strs) output_names.push_back(s.c_str());

    // Run inference
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr},
                                                         input_names.data(), &input_tensor, 1,
                                                         output_names.data(), 1);

    // Extract the result
    float* result = output_tensors.front().GetTensorMutableData<float>();
    return result[0]; // Assuming a single scalar output
}

}  // namespace infer
