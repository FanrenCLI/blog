---
title: Mobilnet(pytorch)
date: 2021-05-08 09:46:42
categories: 
- Deep Learning
tags:
- Pytorch
- MobileNet
author: Fanrencli
---
## 深度学习神经网络特征提取（六）

本次文章给出`MobileNet`的`Pytorch`版本的代码。关于网络的讲解部分，大家参考前期的[文章](http://fanrencli.cn/2021/04/20/feature-extractor/mobilenet/)

### MobileNetv1
```python
class DepthwiseSeparabel(nn.Module):
    def __init__(self,input_channel,output,stride=1):
        super(DepthwiseSeparabel,self).__init__()
        self.depth_wise = nn.Conv2d(input_channel,input_channel,kernel_size=3, stride = stride, padding =1,groups = input_channel)
        self.batch1 = nn.BatchNorm2d(input_channel)
        self.relu1 = nn.ReLU6(inplace = True)
        self.separable = nn.Conv2d(input_channel,output,kernel_size=1,stride = 1)
        self.batch2 = nn.BatchNorm2d(output)
        self.relu2  = nn.ReLU6(inplace = True)
    def forward(self,x):
        x = self.depth_wise(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.separable(x)
        x = self.batch2(x)
        x = self.relu2(x)
        return x

```
```python
class MobileNetv1(nn.Module):
    def __init__(self,num_classes):
        super(MobileNetv1,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride = 2,padding =1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            DepthwiseSeparabel(32,64),
            DepthwiseSeparabel(64,128,2),
            DepthwiseSeparabel(128,128),
            DepthwiseSeparabel(128,256,2),
            DepthwiseSeparabel(256,256),
            DepthwiseSeparabel(256,512,2),
            DepthwiseSeparabel(512,512),
            DepthwiseSeparabel(512,512),
            DepthwiseSeparabel(512,512),
            DepthwiseSeparabel(512,512),
            DepthwiseSeparabel(512,512),
            DepthwiseSeparabel(512,1024,2),
            DepthwiseSeparabel(1024,1024),
        )
        self.avg = nn.AdaptiveMaxPool2d(1)

        self.drop1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(1024,num_classes)
    def forward(self,x):
        x = self.model(x)
        x = self.avg(x)
        x = x.view(x.size(0),x.size(1))
        x = self.drop1(x)
        x = self.linear1(x)
        return x

```

### MobileNetv2
```python
class DepthwiseSeparabel(nn.Module):
    def __init__(self,input_channel,output,stride=1):
        super(DepthwiseSeparabel,self).__init__()
        self.depth_wise = nn.Conv2d(input_channel,input_channel,kernel_size=3, stride = stride, padding =1,groups = input_channel)
        self.batch1 = nn.BatchNorm2d(input_channel)
        self.relu1 = nn.ReLU6(inplace = True)
        self.separable = nn.Conv2d(input_channel,output,kernel_size=1,stride = 1)
        self.batch2 = nn.BatchNorm2d(output)
    def forward(self,x):
        x = self.depth_wise(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.separable(x)
        x = self.batch2(x)
        return x
```
```python
class inverted_res_block(nn.Module):
    def __init__(self,input_channel,output,stride,expansion,first_inverted_res_block = False):
        super(inverted_res_block,self).__init__()
        if not first_inverted_res_block:
            self.model = nn.Sequential(
                nn.Conv2d(input_channel,expansion*input_channel,kernel_size=1,stride = 1),
                nn.BatchNorm2d(expansion*input_channel),
                nn.ReLU6()
            )
        else:
            self.model = nn.Sequential()
        self.depth_wise_separable = DepthwiseSeparabel(input_channel*expansion,output,stride = stride)
    def forward(self,x):
        input_data = x
        x = self.model(x)
        x = self.depth_wise_separable(x)
        if x.shape == input_data.shape:
            x += input_data
        return x
```
```python 
class MobileNetv2(nn.Module):
    def __init__(self,num_classes):
        super(MobileNetv2,self).__init__()
        # input = [3,224,224]
        self.model = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            inverted_res_block(32,16,1,1,True),
            inverted_res_block(16,24,2,6),
            inverted_res_block(24,24,1,6),
            inverted_res_block(24,32,2,6),
            inverted_res_block(32,32,1,6),
            inverted_res_block(32,32,1,6),
            inverted_res_block(32,64,2,6),
            inverted_res_block(64,64,1,6),
            inverted_res_block(64,64,1,6),
            inverted_res_block(64,64,1,6),
            inverted_res_block(64,96,2,6),
            inverted_res_block(96,96,1,6),
            inverted_res_block(96,96,1,6),
            inverted_res_block(96,160,2,6),
            inverted_res_block(160,160,1,6),
            inverted_res_block(160,160,1,6),
            inverted_res_block(160,320,1,6),
            nn.Conv2d(320,1280,kernel_size=1, stride=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(), 
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1280,num_classes)
    def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0),x.size(1))
        x = self.fc(x)
        return x
```

### MobileNetv3

```python 
class hard_swish(nn.Module):
    def __init__(self,inplace=True):
        super(hard_swish,self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self,x):
        x = x*self.relu(x+3.)/6.
        return x
```
```python
class squeeze(nn.Module):
    def __init__(self,up_dim):
        super(squeeze,self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.model = nn.Sequential(
            nn.Linear(up_dim,up_dim//4),
            nn.ReLU6(inplace=True),
            nn.Linear(up_dim//4,up_dim),
            hard_swish(inplace=True)
        )
    def forward(self,x):
        input_data = x
        x = self.avg(x)
        x = x.view(input_data.size(0),input_data.size(1))
        x = self.model(x)
        x = x.view(input_data.size(0),input_data.size(1),1,1)
        return torch.mul(input_data,x)
```
```python
class bottleneck(nn.Module):
    def __init__(self,input_channel,output,kernel_size,stride,up_dim,sq,activation_fun):
        super(bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(input_channel,up_dim,kernel_size=1,stride=1)
        self.batch1 = nn.BatchNorm2d(up_dim)
        self.act_fun1 = hard_swish(inplace=True) if activation_fun == 'HS' else nn.ReLU6(inplace=True)
        self.depth_wise = nn.Conv2d(up_dim,up_dim,kernel_size=kernel_size, stride = stride, padding =(kernel_size-1)//2,groups = up_dim)
        self.batch2 = nn.BatchNorm2d(up_dim)
        self.act_fun2 = hard_swish(inplace=True) if activation_fun == 'HS' else nn.ReLU6(inplace=True)
        self.squeeze = nn.Sequential()
        if sq:
            self.squeeze = squeeze(up_dim)
        self.conv2 = nn.Conv2d(up_dim,output,kernel_size=1,stride = 1)
        self.batch3 = nn.BatchNorm2d(output)
    def forward(self,x):
        input_data = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act_fun1(x)
        x = self.depth_wise(x)
        x = self.batch2(x)
        x = self.act_fun2(x)
        x = self.squeeze(x)
        x = self.conv2(x)
        x = self.batch3(x)
        if x.shape == input_data.shape:
            x +=input_data
        return x
```
```python
class MobileNetv3_small(nn.Module):
    def __init__(self,num_classes):
        super(MobileNetv3_small,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            hard_swish(inplace=True),
            #16,112,112 ->16,56,56
            bottleneck(input_channel=16,output=16,kernel_size=3,stride=2,up_dim=16,sq=True,activation_fun='RE'),
            #16,56,56->24,28,28
            bottleneck(input_channel=16,output=24,kernel_size=3,stride=2,up_dim=72,sq=False,activation_fun='RE'),
            bottleneck(input_channel=24,output=24,kernel_size=3,stride=1,up_dim=88,sq=False,activation_fun='RE'),
            #24,28,28->40,14,14
            bottleneck(input_channel=24,output=40,kernel_size=5,stride=2,up_dim=96,sq=True,activation_fun='HS'),
            bottleneck(input_channel=40,output=40,kernel_size=5,stride=1,up_dim=240,sq=True,activation_fun='HS'),
            bottleneck(input_channel=40,output=40,kernel_size=5,stride=1,up_dim=240,sq=True,activation_fun='HS'),
            #40,14,14->48,14,14
            bottleneck(input_channel=40,output=48,kernel_size=5,stride=1,up_dim=120,sq=True,activation_fun='HS'),
            bottleneck(input_channel=48,output=48,kernel_size=5,stride=1,up_dim=144,sq=True,activation_fun='HS'),
            #48,14,14->96,7,7
            bottleneck(input_channel=48,output=96,kernel_size=5,stride=2,up_dim=288,sq=True,activation_fun='HS'),
            bottleneck(input_channel=96,output=96,kernel_size=5,stride=1,up_dim=576,sq=True,activation_fun='HS'),
            bottleneck(input_channel=96,output=96,kernel_size=5,stride=1,up_dim=576,sq=True,activation_fun='HS'),
            nn.Conv2d(96,576,kernel_size=1,stride=1),
            nn.BatchNorm2d(576),
            hard_swish(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.linear1 = nn.Linear(576,1024)
        self.act_fun1 = hard_swish(inplace=True)
        self.linear2 = nn.Linear(1024,num_classes)
    def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0),x.size(1))
        x = self.linear1(x)
        x = self.act_fun1(x)
        x = self.linear2(x)
        return x
```
至此，`MobileNet`网络的pytorch版本全部更新。
对了，如果有读者想要看一下网络结构的细节部分，此处提供一下代码，此处代码可以进行适当修改，无缝链接到其他文章中的网络。
```python
net = MobileNet(10)
net.to(torch.device('cuda'))
input = torch.randn(10,3,224,224)
out = net(input)
#网络结构
print(net)
#输出参数
print(out.shape)
#网络细节
summary(net,(3,299,299))
```