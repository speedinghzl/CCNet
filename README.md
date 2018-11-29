# CCNet: Criss-Cross Attention for Semantic Segmentation
By [Zilong Huang](http://speedinghzl.github.io), [Xinggang Wang](http://www.xinggangw.info/index.htm), [Lichao Huang](https://scholar.google.com/citations?user=F2e_jZMAAAAJ&hl=en), [Chang Huang](https://scholar.google.com/citations?user=IyyEKyIAAAAJ&hl=zh-CN), [Yunchao Wei](https://weiyc.github.io/), [Wenyu Liu](http://mclab.eic.hust.edu.cn/MCWebDisplay/PersonDetails.aspx?Name=Wenyu%20Liu).

This code is a implementation of the experiments on Cityscapes in the [CCNet](https://github.com/speedinghzl/CCNet). The code is developed based on the Pytorch framework.

We implement our method based on open source [pytorch segmentation toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox). 

## Introduction
![Overview of CCNet](http://pixkzddvl.bkt.gdipper.com/architecture.png)
Overview of the proposed CCNet for semantic segmentation. The proposed recurrent criss-cross attention takes as input feature maps **H** and output feature maps **H''** which obtain rich and dense contextual information from all pixels. Recurrent criss-cross attention module can be unrolled into R=2 loops, in which all Criss-Cross Attention modules share parameters.

## Visualization of the attention map
![Overview of Attention map](http://pixkzddvl.bkt.gdipper.com/attention_vis.png)
To get a deeper understanding of our RCCA, we visualize the learned attention masks as shown in the figure.  For each input image, we select one point (green cross) and show its corresponding attention maps when **R=1** and **R=2** in columns 2 and 3 respectively. In the figure, only contextual information from the criss-cross path of the target point is capture when $R=1$. By adopting one more criss-cross module, ie, **R=2**  the RCCA can finally aggregate denser and richer contextual information compared with that of **R=1**. Besides, we observe that the attention module could capture semantic similarity and long-range dependencies. 

### License

CCNet is released under the MIT License (refer to the LICENSE file for details).

### Citing CCNet

If you find CCNet useful in your research, please consider citing:

    @article{huang2018ccnet,
        title={CCNet: Criss-Cross Attention for Semantic Segmentation},
        author={Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
        booktitle={Arxiv},
        year={2018}
    }
    
### Requirements

To install PyTorch>=0.4.0, please refer to https://github.com/pytorch/pytorch#installation.

### Compiling

Some parts of **InPlace-ABN** and **Criss-Cross Attention** have a native CUDA implementation, which must be compiled with the following commands:
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

## Acknowledgment
The work was mainly done during an internship at [Horizon Robotics](http://en.horizon.ai/).
