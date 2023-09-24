// Copyright (c) 2023 Chuanze Lu
#include "torch_lm_model.h"
#include "torch_lm_inference.h"

auto lm_resource = std::make_shared<lulm::InferenceResoure>();
auto lm_model = std::make_shared<lulm::TorchLmModel>();
auto lm_dict = std::make_shared<lulm::SymbolTable>();


#include "lm_ppl.h"

int lulm_ppl_init(const char* model_dir)
{
    std::string modelDir = std::string(model_dir);
    std::string modelPath = modelDir + "/final.jit";
    std::string dictPath = modelDir + "/lang_char.txt";
    lm_model->Read(modelPath, 4);
    lm_dict->Read(dictPath);
    lm_resource->model = lm_model;
    lm_resource->symbol_table = lm_dict; 

    return 0;
}

float lulm_ppl_compute(const std::string& text)
{
    lulm::TorchLmInference inf =  lulm::TorchLmInference(lm_resource);

    float loss = inf.Ppl(text);
    
    return loss;
}

