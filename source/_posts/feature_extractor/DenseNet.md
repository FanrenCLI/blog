---
title: DenseNet
date: 2021-05-13 12:14:52
top: true
cover: true
categories: 
- Deep Learning
tags:
- DenseNet
- Pytorch
author: Fanrencli
---

## 深度学习神经网络特征提取（八）

### DenseNet网络介绍

前几期文章针对当下流行的特征提取网络做了一些介绍，本章继续前期的特征提取网络的内容介绍一下传说比`ResNet`还要强大的网络——`DenseNet`。首先是`DenseNet`的特点：
- 减轻了梯度消失的情况
- 特征层的重复利用
- 参数量减少

`DenseNet`的网络结构也比较具有特点，通过密集的连接来实现特征的重复利用，网络结构主要如下图：

![DenseNet网络结构](http://39.106.34.39:4567/_20210513122750.png)

`DenseNet`网络中总共包含三个Block，Block之间通过卷积和池化层进行连接实现尺寸的缩减，而每个Block内部则保持特征层的大小尺寸不变，只修改通道数来进行特征提取，这样的实现方式在代码的书写上很方便。

接下来我们就来实现一下`DenseNet`的代码吧，本文给出的`DenseNet`的代码包含有：`DenseNet-121`,`DenseNet-169`,`DenseNet-201``DenseNet-264`。

![DenseNet不同类型的网络细节](http://39.106.34.39:4567/_20210513123403.png)

### 基础层
代码主要对应于dense_block中基础模块，进行特征提取和调整通道数
```python
class conv_block(nn.Module):
    def __init__(self,input_channel,growth_rate):
        super(conv_block,self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel,growth_rate*4,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(growth_rate*4),
            nn.Conv2d(growth_rate*4,growth_rate,kernel_size=3,padding =1,stride=1,bias=False)
        )
    def forward(self,x):
        out = self.model(x)
        return torch.cat([x,out],1)
```

### Dense_Block模块 
对应整个网络中的三个基础块，用于特征提取，网络的主要特征也在这里体现，需要结合基础层理解网络的特点。
```python 
class DenseNet_block(nn.Module):
    def __init__(self,input_channel,blocks,growth_rate):
        super(DenseNet_block,self).__init__()
        self._dense_block = []
        for i in range(blocks):
            self._dense_block.append(conv_block(input_channel+i*growth_rate,growth_rate))
        self._dense_block = nn.ModuleList(self._dense_block)
    def forward(self,x):
        for f in self._dense_block:
            x = f(x)
        return x

```
### 调整层
用于调整特征层的尺寸以及降低一定的通道数
```python 
class transition_block(nn.Module):
    def __init__(self,input_channel,reduction):
        super(transition_block,self).__init__()
        self.model1 = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel,int(math.floor(input_channel*reduction)),kernel_size=1,stride=1,bias=False),
            nn.AvgPool2d(kernel_size=2,stride=2)
        )
    def forward(self,x):
        x = self.model1(x)
        return x

```
### 网络主干结构
```python 
class DenseNet(nn.Module):
    def __init__(self,num_classes,blocks):
        # blocks 对应不同的DenseNet121,DenseNet169，DenseNet201，有[6, 12, 24, 16],[6, 12, 32, 32],[6, 12, 48, 32]
        super(DenseNet,self).__init__()
        # input_shape : 3,224,224
        # 3,224,224 ->64,56,56
        self.model1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,padding=3,stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,padding=1,stride=2)
        )
        # 64,56,56 -> 256=64+32*blocks[0],56,56
        self.dense_block1 = DenseNet_block(input_channel=64,blocks=blocks[0],growth_rate=32)
        # 256,56,56 -> 128,28,28
        self.transition_block1 = transition_block(256,0.5)
        # 128,28,28 -> 512=128+32*block[1],28,28
        self.dense_block2 = DenseNet_block(input_channel=128,blocks=blocks[1],growth_rate=32)
        # 512,28,28 -> 256,14,14
        self.transition_block2 = transition_block(512,0.5)
        # 256,14,14 -> 1024=256+32*blocks[2],14,14
        self.dense_block3 = DenseNet_block(input_channel=256,blocks=blocks[2],growth_rate=32)
        # 1024,14,14 -> 512,7,7
        self.transition_block3 = transition_block(1024,0.5)
        # 512,7,7 -> 1024=512+32*block[3],7,7
        self.dense_block4 = DenseNet_block(input_channel=512,blocks=blocks[3],growth_rate=32)

        self.model2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024,num_classes)
    def forward(self,x):
        x = self.model1(x)
        x = self.dense_block1(x)
        x = self.transition_block1(x)
        x = self.dense_block2(x)
        x = self.transition_block2(x)
        x = self.dense_block3(x)
        x = self.transition_block3(x)
        x = self.dense_block4(x)
        x = self.model2(x)
        x = x.view(x.size(0),x.size(1))
        x = self.fc(x)
        return x
```
下面给出测试代码，注意由于论文中提到对ImageNet数据集测试所用的growth_rate为32，所以本文所用的growth_rate都为32，如有需要自行修改。
```python 
net = DenseNet(10,[6,12,24,16]) #根据对应的层数进行修改
input = torch.randn(2,3,224,224)
out = net(input)
print(net)
print(out.shape)
```


