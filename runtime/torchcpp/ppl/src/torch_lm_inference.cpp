// Copyright (c) 2023 Chuanze Lu
#include "torch_lm_inference.h"

#include <fstream>
#include <vector>
namespace lulm
{
    std::vector<std::string> StringSplit(std::string str)
    {
        int size = str.size();
        std::vector<std::string> ans;
        int j = 0;
        for (int i = 0; i < size; i++)
        {
            if(str[i] == ' ')
            {
                ans.push_back(str.substr(j, i - j));
                j = i + 1;
            }
        }

        ans.push_back(str.substr(j, size - j));

        return ans;
    }

    void SymbolTable::Read(const std::string& dict_file)
    {
        std::ifstream readFile;
        readFile.open(dict_file);

        if(readFile.is_open())
        {
            std::string text_line;
            while(getline(readFile, text_line))
            {
                std::vector<std::string> line_split = StringSplit(text_line);

                int idd = std::stoi(line_split[1]);
                Id2Symbol.insert(make_pair(idd, line_split[0]));
                Symbol2Id.insert(make_pair(line_split[0], idd));
            }
        }

    }

    std::vector<int> SymbolTable::Text2Tokens(const std::string& text)
    {
        std::vector<int> tokens;
        for(int i = 0; i < text.size(); )
        {
            std::string str = text.substr(i, 3);
            tokens.push_back(Symbol2Id[str]);
            i = i + 3;
        }

        return tokens;
    }

    TorchLmInference::TorchLmInference(std::shared_ptr<InferenceResoure> resource)
    :model_(resource->model),
    symbol_table_(resource->symbol_table)
    {

    }

    float TorchLmInference::Loss(std::string text)
    {

        torch::NoGradGuard no_grad;

        std::vector<int> tokens = symbol_table_->Text2Tokens(text);
        int token_len = tokens.size();
        torch::Tensor x = torch::zeros({1, token_len}, torch::kLong);
        
        torch::Tensor row = torch::from_blob(tokens.data(), {token_len}, torch::kInt32);

        x[0] = std::move(row);

        // std::vector<int> lens;
        // lens.push_back(token_len);
        // torch::Tensor x_len = torch::tensor(lens);

        torch::Tensor x_len = torch::zeros({1}, torch::kInt);
        x_len[0] = token_len;

        int sos_id = symbol_table_->Symbol2Id["<sos>"];
        int eos_id = symbol_table_->Symbol2Id["<eos>"];

        std::vector<torch::jit::IValue> inputs = {x,
                                                  x_len,
                                                  sos_id,
                                                  eos_id,
                                                  -1};

        // std::vector<torch::jit::IValue> inputs = {x,
        //                                           x_len
        //                                           };

        auto loss = model_->torch_model()->get_method("inference_loss")(inputs).toTensor();

        float* item = loss.data_ptr<float>();

        std::cout << *item << std::endl;

        return *item;
    }
}