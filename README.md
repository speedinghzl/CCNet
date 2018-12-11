# CCNet: Criss-Cross Attention for Semantic Segmentation
By [Zilong Huang](http://speedinghzl.github.io), [Xinggang Wang](http://www.xinggangw.info/index.htm), [Lichao Huang](https://scholar.google.com/citations?user=F2e_jZMAAAAJ&hl=en), [Chang Huang](https://scholar.google.com/citations?user=IyyEKyIAAAAJ&hl=zh-CN), [Yunchao Wei](https://weiyc.github.io/), [Wenyu Liu](http://mclab.eic.hust.edu.cn/MCWebDisplay/PersonDetails.aspx?Name=Wenyu%20Liu).

This code is a implementation of the experiments on Cityscapes in the [CCNet](https://arxiv.org/abs/1811.11721). 
We implement our method based on open source [pytorch segmentation toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox). 

## Introduction
![motivation of CCNet](http://pixkzddvl.bkt.gdipper.com/motivation.png)
Long-range dependencies can capture useful contextual information to benefit visual understanding problems. In this work, we propose a Criss-Cross Network (CCNet) for obtaining such important information through a more effective and efficient way. Concretely, for each pixel, our CCNet can harvest the contextual information of its surrounding pixels on the criss-cross path through a novel criss-cross attention module. By taking a further recurrent operation, each pixel can finally capture the long-range dependencies from all pixels. Overall, our CCNet is with the following merits: 
- **GPU memory friendly**  
- **High computational efficiency** 
- **The state-of-the-art performance** 

## Architecture
![Overview of CCNet](http://pixkzddvl.bkt.gdipper.com/architecture.png)
Overview of the proposed CCNet for semantic segmentation. The proposed recurrent criss-cross attention takes as input feature maps **H** and output feature maps **H''** which obtain rich and dense contextual information from all pixels. Recurrent criss-cross attention module can be unrolled into R=2 loops, in which all Criss-Cross Attention modules share parameters.

## Visualization of the attention map
![Overview of Attention map](http://pixkzddvl.bkt.gdipper.com/attention_vis.png)
To get a deeper understanding of our RCCA, we visualize the learned attention masks as shown in the figure.  For each input image, we select one point (green cross) and show its corresponding attention maps when **R=1** and **R=2** in columns 2 and 3 respectively. In the figure, only contextual information from the criss-cross path of the target point is capture when **R=1**. By adopting one more criss-cross module, ie, **R=2**  the RCCA can finally aggregate denser and richer contextual information compared with that of **R=1**. Besides, we observe that the attention module could capture semantic similarity and long-range dependencies. 

### License

CCNet is released under the MIT License (refer to the LICENSE file for details).

### Citing CCNet

If you find CCNet useful in your research, please consider citing:

    @article{huang2018ccnet,
        title={CCNet: Criss-Cross Attention for Semantic Segmentation},
        author={Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
        journal = {arXiv preprint arXiv:1811.11721},
        year={2018}
    }
    
### Requirements

To install PyTorch>=0.4.0, please refer to https://github.com/pytorch/pytorch#installation.   
4 x 12G GPUs (_e.g._ TITAN XP)

### Compiling

Some parts of **InPlace-ABN** and **Criss-Cross Attention** have native CUDA implementations, which must be compiled with the following commands:
```bash
cd libs
sh build.sh
python build.py

cd ../cc_attention
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

### Dataset and pretrained model

Plesae download cityscapes dataset and unzip the dataset into `YOUR_CS_PATH`.

Please download MIT imagenet pretrained [resnet101-imagenet.pth](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth), and put it into `dataset` folder.

### Training and Evaluation
```bash
./run_local.sh YOUR_CS_PATH
``` 

### models
We run CCNet with *R=1,2* three times on cityscape dataset separately and report the results in the following table.
Please note there exist some problems about the validation/testing set accuracy gap (1~2%). You need to run multiple times
to achieve a small gap or turn on OHEM flag.

| **R** | **mIOU on cityscape val set**           | **Link** |
|:-------:|:---------------------:|:---------:|
| 1 | 77.22 & **77.91** & 76.89 | [77.91](https://drive.google.com/open?id=13j06I4e50T41j_2HQl4sksrLZihax94L) |
| 2 | **79.74** & 79.22 & 78.40 | [79.74](https://drive.google.com/open?id=1IxXm8qxKmfDPVRtT8uuDNEvSQsNVTfLC) |
| 3 | -              | -    |

## Acknowledgment
The work was mainly done during an internship at [Horizon Robotics](http://en.horizon.ai/).

## Thanks to the Third Party Libs
Self-attention related methods:   
[Object Context Network](https://github.com/PkuRainBow/OCNet)    
[Dual Attention Network](https://github.com/junfu1115/DANet)   
Semantic segmentation toolboxs:   
[pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox)   
[semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)   
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
