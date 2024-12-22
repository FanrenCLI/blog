---
title: DeepLabV3+
date: 2021-05-18 09:38:23
categories:
- Deep Learning
tags:
- semantic segmentation
- DeepLabv3+
author: Fanrencli
---

## 深度学习之语义分割DeepLabV3+(2018)


### DeepLabV3+网络简介

在传统领域的语义分割中，从`FCN`开始提出的全卷积网络，到`DeepLabv3+`网络的提出，基本上语义分割领域中大部分的问题都可以得到解决。在此之后的语义分割领域中出现的文章大部分都是结合了注意力机制提出来的，本文针对传统语义分割的发展过程的新高峰——`DeepLabV3+`进行介绍，针对其本身的优点进行说明。

首先我们来看一下`DeepLabv3+`所作出的优化：
- `DeepLabv3+`是在`DeepLabv3`的编码与解码的基础上，应用了`DeepLabv3`的编码结构，在解码部分进行了改进
- `DeepLabv3+`使用了空洞卷积进行特征提取，能够随意控制特征的提取的分辨率
- `DeepLabv3+`使用了`Xception`作为主干特征提取网络，采用深度可分离卷积和ASPP模块

通过`DeepLabv3+`论文中的阐述，我们可以大致了解整个`DeepLabv3+`的网络结构，如下图：

![DeepLabv3+网络结构](http://fanrencli.cn/fanrencli.cn/_20210523162658.png)

### Xception网络优化

`DeepLabv3+`采用了`Xception`作为主干特征提取的网络，并对其进行了优化改进，改进主要包括两个方面：
- `DeepLabv3+`在`Xception`中引入了空洞卷积
- `DeepLabv3+`对`Xception`的基础块进行了增加，中间块从原来的8次重复加深变成16次重复加深

具体改进后的网络如下图所示：

![DeepLabv3+网络结构](http://fanrencli.cn/fanrencli.cn/_20210523175240.png)

代码展示如下：
```python
class DepthwiseSeparabel(nn.Module):
    def __init__(self,input_channel,output,stride=1, dilation = 1,kernel_size = 1, padding = 0, activate_first = False):
        super(DepthwiseSeparabel,self).__init__()
        self.relu0 = nn.ReLU(inplace=True)
        self.depth_wise = nn.Conv2d(input_channel,input_channel,kernel_size=kernel_size, stride = stride, padding =padding,dilation =dilation, groups = input_channel,bias =False)
        self.batch1 = nn.BatchNorm2d(input_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.separable = nn.Conv2d(input_channel,output,kernel_size=1,stride = 1,bias =False)
        self.batch2 = nn.BatchNorm2d(output)
        self.relu2 = nn.ReLU(inplace=True)
        self.activate_first = activate_first
    def forward(self,x):
        if self.activate_first:
            x = self.relu0(x)
        x = self.depth_wise(x)
        x = self.batch1(x)
        if not self.activate_first:
            x = self.relu1(x)
        x = self.separable(x)
        x = self.batch2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x

```
```python
class Xception_Entry_flow(nn.Module):
    def __init__(self,input_channel,output_channel,stride=1):
        super(Xception_Entry_flow,self).__init__()
        self.shortcut = nn.Conv2d(input_channel,output_channel,1,stride=stride, bias=False)
        self.shortcutbn = nn.BatchNorm2d(output_channel)
        self.hook_layer = None
        self.sepconv1 = DepthwiseSeparabel(input_channel,output_channel,kernel_size = 3,stride=1,padding=1,dilation=1,activate_first=True)
        self.sepconv2 = DepthwiseSeparabel(output_channel,output_channel,kernel_size = 3,stride=1,padding=1,dilation=1,activate_first=True)
        self.sepconv3 = DepthwiseSeparabel(output_channel,output_channel,kernel_size = 3,stride=stride,padding=1,dilation=1,activate_first=True)
    def forward(self,x):
        shortcut = self.shortcut(x)
        shortcut = self.shortcutbn(shortcut)
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        self.hook_layer = x
        x = self.sepconv3(x)
        x+=shortcut
        return x
```
```python
class Xception_Middle_flow(nn.Module):
    def __init__(self,input_channel,atrous):
        super(Xception_Middle_flow,self).__init__()
        self.sepconv1 = DepthwiseSeparabel(input_channel,input_channel,kernel_size =3,stride=1,padding=atrous,dilation=atrous,activate_first=True)
        self.sepconv2 = DepthwiseSeparabel(input_channel,input_channel,kernel_size =3,stride=1,padding=atrous,dilation=atrous,activate_first=True)
        self.sepconv3 = DepthwiseSeparabel(input_channel,input_channel,kernel_size =3,stride=1,padding=atrous,dilation=atrous,activate_first=True)
    def forward(self,x):
        skip = x
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.sepconv3(x)
        x+=skip
        return x
```
```python
class Xception_Exit_flow(nn.Module):
    def __init__(self,in_filters,out_filters,strides=1,atrous=1):
        super(Xception_Exit_flow,self).__init__()
        self.shortcut = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
        self.shortcutbn = nn.BatchNorm2d(out_filters)
        self.hook_layer = None
        self.sepconv1 = DepthwiseSeparabel(in_filters,in_filters,kernel_size =3,stride=1,padding=atrous,dilation=atrous,activate_first=True)
        self.sepconv2 = DepthwiseSeparabel(in_filters,out_filters,kernel_size =3,stride=1,padding=atrous,dilation=atrous,activate_first=True)
        self.sepconv3 = DepthwiseSeparabel(out_filters,out_filters,kernel_size =3,stride=1,padding=atrous,dilation=atrous,activate_first=True)

    def forward(self,x):
        shortcut = self.shortcut(x)
        shortcut = self.shortcutbn(shortcut)

        x = self.sepconv1(x)
        x = self.sepconv2(x)
        self.hook_layer = x
        x = self.sepconv3(x)

        x+=shortcut
        return x
```
```python
class Xception(nn.Module):
    def __init__(self,os):
        super(Xception,self).__init__()
        if os == 8:
            stride_list = [2,1,1]
        elif os == 16:
            stride_list = [2,2,1]
        # input shape  = 512,512,3
        self.first_block = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=2, padding = 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.Entry_flow0 = Xception_Entry_flow(64,128,2)
        self.Entry_flow1 = Xception_Entry_flow(128,256,stride_list[0])
        self.Entry_flow2 = Xception_Entry_flow(256,728,stride_list[1])
        rate = 16//os
        self.Middle_flow = nn.Sequential(
            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),

            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),

            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),

            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate),
            Xception_Middle_flow(728,atrous = rate)
        )
        self.Exit_flow = Xception_Exit_flow(728,1024,stride_list[2],atrous=rate)

        self.conv0 = DepthwiseSeparabel(1024,1536,kernel_size = 3,stride = 1,padding = rate,dilation=rate,activate_first=False)
        self.batch0 = nn.BatchNorm2d(1536)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = DepthwiseSeparabel(1536,1536,kernel_size = 3,stride = 1,padding = rate,dilation=rate,activate_first=False)
        self.batch1 = nn.BatchNorm2d(1536)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparabel(1536,2048,kernel_size = 3,stride = 1,padding = rate,dilation=rate,activate_first=False)
        self.batch2 = nn.BatchNorm2d(2048)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.first_block(x)
        x = self.Entry_flow0(x)
        x = self.Entry_flow1(x)
        x = self.Entry_flow2(x)
        skip = self.Entry_flow1.hook_layer
        x = self.Middle_flow(x)
        x = self.Exit_flow(x)

        x = self.conv0(x)
        x = self.batch0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        return x,skip
```

以上就是`DeepLabv3+`针对`Xception`作出的所有改进，主干网络输出为一个4倍下采样的特征和最后的特征层，接着后面就是`DeepLabv3+`的解码部分
### DeepLabV3解码部分

`DeepLabv3+`的解码部分主要就是将主干网络输出的两个特征层进行处理，对4倍下采样的特征进行通道调整，最后的特征层进行ASPP模块处理以及上采样，然后将两个特征层进行融合后再进行上采样，具体代码如下：

```python
class ASPP(nn.Module):
    def __init__(self,input_channel,output_channel,atrous_rate):
        super(ASPP,self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv0 = nn.Conv2d(input_channel,output_channel,kernel_size=1,stride=1,bias=False)
        self.batch0 = nn.BatchNorm2d(output_channel)
        self.relu0 = nn.ReLU(inplace=True)
        # resize the globalFeature
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=1,stride=1,bias =False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel,kernel_size = 3, stride=1, padding=atrous_rate[0], dilation=atrous_rate[0],bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),	
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel,kernel_size = 3, stride=1, padding=atrous_rate[1], dilation=atrous_rate[1],bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),	
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel,kernel_size = 3, stride=1, padding=atrous_rate[2], dilation=atrous_rate[2],bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),	
        )
        self.conv1 = nn.Conv2d(output_channel*5, output_channel,kernel_size=1,stride=1)
        self.batch1 = nn.BatchNorm2d(output_channel)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self,x):
        [b,c,h,w] = x.size()
        global_features = self.avg(x)
        global_features = self.conv0(global_features)
        global_features = self.batch0(global_features)
        global_features = self.relu0(global_features)
        global_features = F.interpolate(global_features,(h,w),None,'bilinear',True)

        p1 = self.branch1(x)
        p2 = self.branch2(x)
        p3 = self.branch3(x)
        p4 = self.branch4(x)
        x = torch.cat([global_features,p1,p2,p3,p4],dim=1)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        return x
```

```python
class Deeplabv3plus(nn.Module):
    def __init__(self,os,num_classes):
        super(Deeplabv3plus,self).__init__()
        if os == 16:
            atrous_rate = [6,12,18]
        else:
            atrous_rate = [12,24,36]
        self.Xception = Xception(os)
        self.aspp = ASPP(2048,256,atrous_rate)
        self.dropout0 = nn.Dropout(0.5)
        self.upsample0 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.conv0 = nn.Conv2d(256,48,kernel_size=1,stride=1,padding = 0,bias=False)
        self.batch0 = nn.BatchNorm2d(48)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(256+48,256,kernel_size=3,stride=1,padding=1,bias=False)
        self.batch1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=False)
        self.batch2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.1)

        self.last_conv = nn.Conv2d(256,num_classes,kernel_size=1,stride=1,padding=0,bias=False)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=4)
    def forward(self,x):
        x,skip  = self.Xception(x)
        x = self.aspp(x)
        x = self.dropout0(x)
        x = self.upsample0(x)
        # 处理skip
        skip = self.conv0(skip)
        skip = self.batch0(skip)
        skip = self.relu0(skip)

        x = torch.cat([x,skip],dim=1)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.last_conv(x)
        x = self.upsample1(x)
        return x

```
至此关于`DeepLabv3+`的主要代码全部给出，至于其他的学习的超参数需要读者自行调整。