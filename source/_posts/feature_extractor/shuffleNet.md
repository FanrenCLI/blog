---
title: ShuffleNetV2
date: 2021-05-15 10:18:02
top: true
cover: true
categories: 
- Deep Learning
tags:
- ShuffleNetV2
- Pytorch
author: Fanrencli
---

## 深度学习神经网络特征提取（九）

### ShuffleNetV2网络介绍

在之前的文章中我们介绍了轻量级的特征提取网络——`MobileNet`系列，从`MobileNetv1`到`MobileNetv3`从论文中的说法来看，是在不断提升的。但是在实际的应用中我们还是要综合考虑，因为用于提取特征的网络所适用的范围是不同的，并不存在万能的特征提取网络，也不一定说最新的网络比过去的网络一定更好，所以在实际的应用中，使用者需要自行判断。

接续之前已经介绍过的轻量级特征提取网络，本文给出又一个较为流行的轻量级特征提取网络——`ShuffleNetV2`。首先介绍一下`ShuffleNetV2`的两个基本常用模块，一个步长为1，所以进行残差连接不需要进行降维，而当步长为2时，则需要对连接的通道进行降维。具体情况可以看下图：

![ShuffleNetV2常用模块](http://39.106.34.39:4567/_20210517165212.png)

了解了`ShuffleNetV2`的基本模块之后，我们来看看`ShuffleNetV2`的具体的网络结构。其中`ShuffleNetV2`分为四种模式，根据输出的通道判断属于那种形式。总体结构主要为以下几个流程：
- 输入（3，224，224）大小的图像进行一次卷积一次池化进行降维，变为（3，56，56）
- 然后进入`ShuffleNetV2`的三个阶段进行特征提取
- 最后一次卷积进行通道数调整，然后平均池化和全连接输出

![ShuffleNetV2网络总体结构](http://39.106.34.39:4567/_20210517165727.png)

代码实现：
```python
#由于进行残差连接只是简单的堆叠，所以对于通道之间的信息交流缺少，通过特征重组可以解决
def channel_shuffle(x,group=2):
    batch_size, num_channels, height, width =x.size()
    assert num_channels % group == 0
    x = x.view(batch_size, group, num_channels // group, height, width)

    x = x.transpose(1,2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x
class BasicUnit(nn.Module):
    def __init__(self, inplanes, outplanes, c_tag=0.5, groups=2):
        super(BasicUnit, self).__init__()
        self.left_part = round(c_tag * inplanes)
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part
        self.conv1 = nn.Conv2d(self.right_part_in, self.right_part_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.right_part_out)
        self.conv2 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=3, padding=1, bias=False,
                               groups=self.right_part_out)
        self.bn2 = nn.BatchNorm2d(self.right_part_out)
        self.conv3 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.right_part_out)
        self.activation = nn.ReLU(inplace=True)

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.groups = groups

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        out = self.conv1(right)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        if self.inplanes == self.outplanes:
            out += right
        return channel_shuffle(torch.cat((left, out), 1), self.groups)
class DownsampleUnit(nn.Module):
    def __init__(self, inplanes, groups=2):
        super(DownsampleUnit, self).__init__()

        self.conv1r = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn1r = nn.BatchNorm2d(inplanes)
        self.conv2r = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False, groups=inplanes)
        self.bn2r = nn.BatchNorm2d(inplanes)
        self.conv3r = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn3r = nn.BatchNorm2d(inplanes)

        self.conv1l = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False, groups=inplanes)
        self.bn1l = nn.BatchNorm2d(inplanes)
        self.conv2l = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn2l = nn.BatchNorm2d(inplanes)
        self.activation = nn.ReLU(inplace=True)

        self.groups = groups
        self.inplanes = inplanes

    def forward(self, x):
        out_r = self.conv1r(x)
        out_r = self.bn1r(out_r)
        out_r = self.activation(out_r)

        out_r = self.conv2r(out_r)
        out_r = self.bn2r(out_r)

        out_r = self.conv3r(out_r)
        out_r = self.bn3r(out_r)
        out_r = self.activation(out_r)

        out_l = self.conv1l(x)
        out_l = self.bn1l(out_l)

        out_l = self.conv2l(out_l)
        out_l = self.bn2l(out_l)
        out_l = self.activation(out_l)

        return channel_shuffle(torch.cat((out_r, out_l), 1))
class ShuffleNetV2(nn.Module):
    def __init__(self, scale=1.0, in_channels=3, num_classes=10):
        super(ShuffleNetV2, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.num_classes = num_classes

        self.num_of_channels = {0.5: [24, 48, 96, 192, 1024], 1: [24, 116, 232, 464, 1024],
                                1.5: [24, 176, 352, 704, 1024], 2: [24, 244, 488, 976, 2048]}
        self.c = self.num_of_channels[scale]
        self.n = [3, 7, 3]
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.shuffles = self._make_shuffles()

        self.conv_last = nn.Conv2d(self.c[-2], self.c[-1], kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.c[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.c[-1], self.num_classes)

    def _make_stage(self, inplanes, outplanes, n, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit{}".format(stage)

        # First module is the only one utilizing stride
        first_module = DownsampleUnit(inplanes=inplanes)
        modules["DownsampleUnit"] = first_module
        second_module = BasicUnit(inplanes=inplanes * 2, outplanes=outplanes )
        modules[stage_name + "_{}".format(0)] = second_module
        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = BasicUnit(inplanes=outplanes, outplanes=outplanes )
            modules[name] = module

        return nn.Sequential(modules)

    def _make_shuffles(self):
        modules = OrderedDict()
        stage_name = "ShuffleConvs"

        for i in range(len(self.c) - 2):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i], stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.shuffles(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # average pooling layer
        x = self.avgpool(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

至此本文给出了`ShuffleNetv2`的所有代码，其中所有的代码按照最简洁的方式给出，其中可能还有其他例如通过SELayer进行特征融合，设置channel split的阈值等等其他的细节参数，本文没有过多设计，如果读者需要，可在看懂本文代码的基础上自行添加。
