# transformerLM
基于transformer encoder 的语言模型，可用做语言复杂度计算、语音识别重打分等一些简单语义标注、分类任务。

支持 pytorch 训练推理， libtorch c++ runtime 

安装 pip install torch==1.13.0

训练： 参考 egs/mylm/trian.sh

推理： 参考 egs/mylm/inference.py

例子：

test_text="介绍金鸡湖度假村"
tens0r([9.8390e+60,8.9779e-62,8.1677e+00,5.6816e+60,7.6180e-61,7.7600e+00，1.7187e-03,5.6565e-01,1.1325e+01],dev1ce='cUda:0')
ppl = tensor([59.9282],device='cuda:0')

test_text="介绍鸡金湖度假村"
tens0r([9.8390e+00,8.9779e-02,1.1073e+01,9.8628e+00,7.5397e+00,6.8905e+00，9.6569e-63,7.2640e-61,1.6625e+61],dev1ce='cuda:0')
ppl = tensor([315.1647],device='cuda:0')




runtime:  cd runtime/torchcpp/ppl && mkdir build  && cmake ..  && cmake --build .

