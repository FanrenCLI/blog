---
title: FCN
date: 2021-05-10 15:31:42
top: true
cover: true
categories:
- Deep Learning
tags:
- semantic segmentation
- FCN
author: Fanrencli
---

## 深度学习之语义分割FCN(2015)

### FCN网络简介
emmmmmmm...时隔几个星期，语义分割系列又开始了，从这期开始所有的代码都会以`pytorch`框架给出。加油吧，少年~~

话不多说，开始今天的语义分割之`FCN`。总的来说，`FCN`网络基本上可以算是语义分割的重量级人物。基于CNN网络在进行卷积和池化的过程中会不断缩小特征层，不可避免丢失了一些图像细节，所以到最后的特征层基本就无法判断每个像素具体属于哪个物体，没有办法做到精确分割。而针对这个问题，`FCN`应运而生。

相比于之前CNN网络，`FCN`网络如同它的名称一般，是一个全卷积的网络，FCN抛弃了传统CNN网络最后的全连接层，全部采用卷积层替换。这样最后获得是一个二维的特征层，便于后面的反卷积扩张，具体情况如下图（根据论文中的阐述，主干网络使用的是VGG16）。
![FCN网络结构](http://39.105.26.229:4567/_20210510164538.png)

经过上图所示的特征提取网络之后，对相应的特征层进行反卷积上采样，将特征层扩大到原来图像的大小，然后计算loss。这样`FCN`的基本结构就完成了。而这里的重点就是在于选择那个特征层进行上采样，由此根据选择的特征层，将FCN网络分成三种形式：`FCN32s`、`FCN16s`和`FCN8s`，分别对应32步长上采样，16步长上采样和8倍步长上采样。具体形式如下图。

![FCN网络结构](http://39.105.26.229:4567/_20210510165258.png)

至于loss函数部分，就是交叉熵函数。接下来代码部分，会给出三种形式的`FCN`代码，以及相关的关键注释。

### FCN32s
```python
class VGG16(nn.Module):
    def __init__(self,num_classes):
        super(VGG16,self).__init__()
        #input_shape(3,500,500) 
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,padding =100,stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding =1,stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride = 2)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding =1,stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128,128,kernel_size=3,padding =1,stride=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv5 = nn.Conv2d(128,256,kernel_size=3,padding =1,stride=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256,256,kernel_size=3,padding =1,stride=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256,256,kernel_size=3,padding =1,stride=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv8 = nn.Conv2d(256,512,kernel_size=3,padding =1,stride=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride = 2)

        self.conv11 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu13 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2,stride = 2)

        self.conv14 = nn.Conv2d(512,4096,7)
        self.relu14 =nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d()

        self.conv15 = nn.Conv2d(4096,4096,1)
        self.relu15 =nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096,num_classes,1)
        self.upscore32 = nn.ConvTranspose2d(num_classes,num_classes,64,stride=32,bias=False)

    def forward(self,x):
        h = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.maxpool3(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.maxpool4(x)

        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        x = self.relu13(x)
        x = self.maxpool5(x)

        x = self.conv14(x)
        x = self.relu14(x)
        x = self.dropout1(x)
        x = self.conv15(x)
        x = self.relu15(x)
        x= self.dropout2(x)

        x = self.score_fr(x)
        x = self.upscore32(x)
        x = x[:, :, 6:6 + h.size()[2], 6:6 + h.size()[3]].contiguous()
        return x
```

### FCN16s

```python
class VGG16(nn.Module):
    def __init__(self,num_classes):
        super(VGG16,self).__init__()
        #input_shape(3,500,500)
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,padding =100,stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding =1,stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride = 2)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding =1,stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128,128,kernel_size=3,padding =1,stride=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv5 = nn.Conv2d(128,256,kernel_size=3,padding =1,stride=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256,256,kernel_size=3,padding =1,stride=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256,256,kernel_size=3,padding =1,stride=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv8 = nn.Conv2d(256,512,kernel_size=3,padding =1,stride=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride = 2)

        self.conv11 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu13 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2,stride = 2)

        self.conv14 = nn.Conv2d(512,4096,7)
        self.relu14 =nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d()

        self.conv15 = nn.Conv2d(4096,4096,1)
        self.relu15 =nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096,num_classes,1)

        self.upscore16 = nn.ConvTranspose2d(num_classes,num_classes,32,stride=16,bias=False)
        self.upscore2 = nn.ConvTranspose2d(num_classes,num_classes,4,stride=2,bias=False)
        self.score_pool4 = nn.Conv2d(512,num_classes,kernel_size=1,stride=1,bias=False)
    def forward(self,x):
        h = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.maxpool3(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.maxpool4(x)
        pool4 = x #1/16  31
        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        x = self.relu13(x)
        x = self.maxpool5(x)

        x = self.conv14(x)
        x = self.relu14(x)
        x = self.dropout1(x)
        x = self.conv15(x)
        x = self.relu15(x)
        x= self.dropout2(x)

        x = self.score_fr(x)

        x = self.upscore2(x) #1/16
        pool4 = self.score_pool4(pool4) #pool4调整通道数
        x = x + pool4[:,:,5:5+x.size(2),5:5+x.size(3)]
        x = x.contiguous()
        x = self.upscore16(x)
        x = x[:,:,27:27+h.size(2),27:27+h.size(3)].contiguous()
        return x
```

### FCN8s
```python
class VGG16(nn.Module):
    def __init__(self,num_classes):
        super(VGG16,self).__init__()
        #input_shape(3,500,500) 
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,padding =100,stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding =1,stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride = 2)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding =1,stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128,128,kernel_size=3,padding =1,stride=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv5 = nn.Conv2d(128,256,kernel_size=3,padding =1,stride=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256,256,kernel_size=3,padding =1,stride=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256,256,kernel_size=3,padding =1,stride=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv8 = nn.Conv2d(256,512,kernel_size=3,padding =1,stride=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride = 2)

        self.conv11 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu13 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2,stride = 2)

        self.conv14 = nn.Conv2d(512,4096,7)
        self.relu14 =nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d()

        self.conv15 = nn.Conv2d(4096,4096,1)
        self.relu15 =nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096,num_classes,1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)


        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)
        self.upscore2x2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
    def forward(self,x):
        h = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.maxpool3(x)
        pool3 = x
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.maxpool4(x)
        pool4 = x
        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        x = self.relu13(x)
        x = self.maxpool5(x)

        x = self.conv14(x)
        x = self.relu14(x)
        x = self.dropout1(x)
        x = self.conv15(x)
        x = self.relu15(x)
        x= self.dropout2(x)

        x = self.score_fr(x)
        #2倍反卷积
        x = self.upscore2(x)
        # 调整pool4通道
        pool4 = self.score_pool4(pool4)
        # 融合 pool4
        x = x + pool4[:, :, 5:5 + x.size(2), 5:5 + x.size(3)]
        # 再次2倍反卷积
        x = self.upscore2x2(x)
        #调整pool3通道数
        pool3 = self.score_pool3(pool3)
        #融合pool3
        x = x + pool3[:, :, 9:9 + x.size(2), 9:9 + x.size(3)]
        # 8倍反卷积
        x = self.upscore8(x)
        x = x[:, :, 31:31 + h.size(2), 31:31 + h.size(3)].contiguous()

        return x
```

至此，关于FCN网络的结构更新完毕了，其中我们输入的原始图片大小为500x500，所以读者在自行构建数据集时需要将图片固定到500x500的大小，如果读者想要使用其他尺寸大小的图片进行训练，则需要修改网络中的一些参数，其中注意本文网络在第一次卷积层中给出了`100`大小的padding，这个主要是为了适合500pixel图片所设计的，如果读者想要自己设计图片大小，则需要自行修改。其中由于在进行倍数反卷积时会导致层数之间的size大小不同，所以在每层融合的时候都进行了一定量的裁剪操作，读者需要考虑到这些，如果你都了解了，那么你就可以自行修改了。
至于FCN的loss函数部分，由于只是一个简单的交叉熵，此处不再进行更新，会在更新完所有语义分割网络之后，进行单独讲解，或者给出一般性的代码。