// Copyright (c) 2023 Chuanze Lu
#ifndef TORCH_LM_INFERENCE_H_
#define TORCH_LM_INFERENCE_H_

#include "torch_lm_model.h"
#include  <map>
namespace lulm {

    class SymbolTable
    {
        public:

        void Read(const std::string& dict_file);

        std::vector<int> Text2Tokens(const std::string& text);

        std::map<int, std::string> Id2Symbol;
        std::map<std::string, int> Symbol2Id;

    };

    struct InferenceResoure{
        std::shared_ptr<TorchLmModel> model = nullptr;
        std::shared_ptr<SymbolTable> symbol_table = nullptr;
    };

    class TorchLmInference
    {
        public:
        TorchLmInference(std::shared_ptr<InferenceResoure> resource);

        float Ppl(std::string text);

        std::shared_ptr<TorchLmModel> model_;
        std::shared_ptr<SymbolTable> symbol_table_;

    };
}

#endif //TORCH_LM_INFERENCE_H_