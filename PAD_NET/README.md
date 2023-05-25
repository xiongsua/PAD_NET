
##Remote Image Dehazing Using Progressive Frequency Filtering and Adaptive Self-Attention

## Abstract
Remote Sensing (RS) dehazing is a very difficult task due to the complex image degradation and haze distribution. Current learning-based methods make great performance for RS dehazing, while they ignore the frequency characteristic and treat various features equally. In this paper, we propose a stage-wise PADehazeNet using the Progressive frequency filtering and Adaptive self-attention mechanism. To be specific, a progressive frequency filtering structure is presented to decompose the high frequency features from the low rank ones. The progressive feature sharing tensors are used to concatenate and share the multi-scale high frequency features. In addition, we apply an adaptive self-attention mechanism to pick the effective features from the low frequency components, so that the global tone can be reconstructed in an efficient way. Extensive experimental results state that the proposed PADehazeNet achieves outstanding performance than the recent comparing models on both synthetic and real-world datasets in the remote sensing dehazing.


## Dependencies

- Linux (Tested on Ubuntu 18.04)
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 0.4.1](https://pytorch.org/): `conda install pytorch=0.4.1 torchvision cudatoolkit=9.2 -c pytorch`
- numpy: `conda install numpy`
- matplotlib: `conda install matplotlib`
- opencv: `conda install opencv`
- imageio: `conda install imageio`
- skimage: `conda install scikit-image`
- tqdm: `conda install tqdm`


##  DATA

[SateHaze1k]:https://aistudio.baidu.com/aistudio/datasetdetail/134292

##pretain
[pre-tian models]:RUL: https://pan.baidu.com/s/1b1k7xfzYMiRQJsWDIsIraA PASS: enj3 
```
@article{bai2022self,
    title = {Remote Image Dehazing Using Progressive Frequency Filtering and Adaptive Self-Attention},
    author = {Shuai Xiong, Yufeng Huang},
   

```
