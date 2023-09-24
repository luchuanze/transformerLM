// Copyright (c) 2023 Chuanze Lu
#ifndef LULM_PPL_H_
#define LULM_PPL_H_

#include <string>

int lulm_ppl_init(const char* model_dir);

float lulm_ppl_compute(const std::string& text);

#endif //LULM_PPL_H_