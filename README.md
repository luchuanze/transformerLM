# fransformerLM
基于transformer encoder 的语言模型，可用做语言复杂度计算、语音识别重打分等一些简单语义标注、分类任务。

支持 pytorch 训练推理， libtorch c++ runtime 

安装 pip install torch==1.13.0

训练： 参考 egs/mylm/trian.sh

推理： 参考 egs/mylm/inference.py

runtime:  cd ppl; mkdir build; cmake ..; cmake --build .
