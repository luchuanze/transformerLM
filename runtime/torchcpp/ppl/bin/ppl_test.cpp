// Copyright (c) 2023 Chuanze Lu
#include "lm_ppl.h"
#include <string>
#include <iostream>

int main(int argc, char *argv[])
{
    const char* model_dir = "/home/cipan/lulmperplexity/runtime/torchcpp/ppl/build/model";

    lulm_ppl_init(model_dir);

    float loss = lulm_ppl_compute("我是中国人");

    std::cout << "sentence ppl is " << loss << std::endl;


    return 0;
}