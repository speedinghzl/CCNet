# Pytorch-segmentation-toolbox [DOC](https://weiyc.github.io/assets/pdf/toolbox.pdf)
Pytorch code for semantic segmentation. This is a minimal code to run PSPnet and Deeplabv3 on Cityscape dataset.
Shortly afterwards, the code will be reviewed and reorganized for convenience.

### Highlights of Our Implementations
- Synchronous BN
- Fewness of Training Time
- Better Reproduced Performance

### Requirements && Install
Python 3.7

4 x 12g GPUs (e.g. TITAN XP)

```bash
# Install **Pytorch-1.1**
$ conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# Install **Apex**
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install **Inplace-ABN**
$ git clone https://github.com/mapillary/inplace_abn.git
$ cd inplace_abn
$ python setup.py install
```

### Dataset and pretrained model

Plesae download cityscapes dataset and unzip the dataset into `YOUR_CS_PATH`.

Please download MIT imagenet pretrained [resnet101-imagenet.pth](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth), and put it into `dataset` folder.

### Training and Evaluation
```bash
./run_local.sh YOUR_CS_PATH [pspnet|deeplabv3] 40000 769,769 0
``` 

### Benefits
Some recent projects have already benefited from our implementations. For example, [CCNet: Criss-Cross Attention for semantic segmentation](https://github.com/speedinghzl/CCNet) and [Object  Context  Network(OCNet)](https://github.com/PkuRainBow/OCNet) currently  achieve  the  state-of-the-art  resultson  Cityscapes  and  ADE20K. In  addition, Our code also make great contributions to [Context Embedding with EdgePerceiving (CE2P)](https://github.com/liutinglt/CE2P), which won the 1st places in all human parsing tracks in the 2nd LIP Challange. 

### Citing

If you find this code useful in your research, please consider citing:

    @misc{huang2018torchseg,
      author = {Huang, Zilong and Wei, Yunchao and Wang, Xinggang, and Liu, Wenyu},
      title = {A PyTorch Semantic Segmentation Toolbox},
      howpublished = {\url{https://github.com/speedinghzl/pytorch-segmentation-toolbox}},
      year = {2018}
    }

### Thanks to the Third Party Libs
[inplace_abn](https://github.com/mapillary/inplace_abn) - 
[Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab) - 
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
