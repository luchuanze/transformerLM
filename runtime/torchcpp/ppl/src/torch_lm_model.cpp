// Copyright (c) 2023 Chuanze Lu
#include "torch_lm_model.h"

namespace lulm
{

    void TorchLmModel::Read(const std::string &model_path, int num_threads)
    {
        torch::jit::script::Module model = torch::jit::load(model_path);

        module_ = std::make_shared<TorchModule>(std::move(model));

        // at::set_num_threads(num_threads);

        torch::NoGradGuard no_grad;

        module_->eval();
    }
}