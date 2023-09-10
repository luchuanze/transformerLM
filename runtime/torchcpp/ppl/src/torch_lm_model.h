// Copyright (c) 2023 Chuanze Lu
#ifndef TORCH_LM_MODEL_H_
#define TORCH_LM_MODEL_H_

#include <memory>
#include <string>
#include <iostream>

#include "torch/script.h"

namespace lulm{

    using TorchModule = torch::jit::script::Module;

    class TorchLmModel
    {
        public:
        TorchLmModel() = default;

        void Read(const std::string& model_path, int num_threads = 1);

        std::shared_ptr<TorchModule> torch_model() const {return module_;}

        private:
        std::shared_ptr<TorchModule> module_ = nullptr;
    };

}


#endif