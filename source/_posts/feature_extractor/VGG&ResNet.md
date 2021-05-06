---
title: VGG & ResNet
date: 2021-05-06 16:26:39
top: true
cover: true
categories: 
- Deep Learning
tags:
- Pytorch
- VGG16
- ResNet101
author: Fanrencli
---
## 深度学习神经网络特征提取（三）

本文接续前文提到的主干特征提取网络，前文的网络构建主要是基于`keras`框架构建的，而在深度学习领域`pytorch`是当前最流行的框架，且深受顶会论文学者的喜爱。因此，在前文已经给出的`keras`代码的基础上，从这篇文章开始会针对前文网络重新用`pytorch`构建。

再次给出`pytorch`代码，还需要注意的是：不同于`keras`，`pytorch`输入的shape=（3，224，224），通道数在前。
### VGG16
```python
class VGG16(nn.Module):
    def __init__(self,input_channel,num_classes):
        super(VGG16,self).__init__()
        #input_shape(3,224,224)
        self.conv1 = nn.Conv2d(input_channel,64,kernel_size=3,padding =1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding =1,stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride = 2)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding =1,stride=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128,128,kernel_size=3,padding =1,stride=1)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv5 = nn.Conv2d(128,256,kernel_size=3,padding =1,stride=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256,256,kernel_size=3,padding =1,stride=1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256,256,kernel_size=3,padding =1,stride=1)
        self.relu7 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel=2,stride=2)

        self.conv8 = nn.Con2d(256,512,kernel_size=3,padding =1,stride=1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu10 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride = 2)

        self.conv11 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu12 = nn.ReLU()
        self.conv13 = nn.Conv2d(512,512,kernel_size=3,padding =1,stride=1)
        self.relu13 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2,stride = 2)

        self.linear1 = nn.Linear(512*7*7,4096)
        self.relu14 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2= nn.Linear(4096,4096)
        self.relu15 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear3= nn.Linear(4096,num_classes)
    def forward(self,x):
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

        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.relu14(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu15(x)
        x = self.dropout2(x)
        x =self.linear3(x)
        return x
```

### ResNet101
```python
class Conv_block(nn.Module):
    def __init__(self,input_channel,filters,strides=2):
        super(Conv_block,self).__init__()
        self.conv1 = nn.Conv2D(input_channel,filters[0],kernel_size=1,strides = strides,bias=True)
        self.batch1 = nn.BatchNorm2d(filters[0])
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(filters[0],filters[1],kernel_size=3,strides=1,padding =1,bias =True)
        self.batch2 = nn.BatchNorm2d(filters[1])
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(filters[1],filters[2],kernel_size=1,strides=1,padding =1,bias =True)
        self.batch3 = nn.BatchNorm2d(filters[2])

        self.conv4 = nn.Conv2d(input_shape,filters[2],kernel_size=1,strides=strides,padding =1,bias =True)
        self.batch4 = nn.BatchNorm2d(filters[2])

        slef.relu3 = nn.ReLU()
    def forward(slef,x):
        shortcut = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        shortcut = self.conv4(shortcut)
        shortcut = self.batch4(shortcut)
        x +=shortcut
        x = self.relu3(x)
        return x
class Identity_block(nn.Module):
    def __init__(self,input_channel,filters):
        super(Identity_block,self).__init__()
        self.conv1 = nn.Conv2D(input_channel,filters[0],kernel_size=1,padding=1,strides=1,bias=True)
        self.batch1 = nn.BatchNorm2d(filters[0])
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2D(filters[0],filters[1],kernel_size=3,padding=1,strides=1,bias=True)
        self.batch2 = nn.BatchNorm2d(filters[1])
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2D(filters[1],filters[2],kernel_size=1,padding=1,strides=1,bias=True)
        self.batch3 = nn.BatchNorm2d(filters[2])

        self.relu3 = nn.ReLU()
    
    def forward(self,x):
        shortcut = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x +=shortcut
        x = self.relu3(x)
        return x
class ResNet101(nn.Module):
    def __init__(self,num_classes):
        super(ResNet101,self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2D(3,64,kernel_size=7,padding=3,strides=2,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,strides=2,padding=1)
            
        )
        self.model2 = nn.Sequential(
            Conv_block(64,[64,64,256],strides=1),
            Identity_block(256,[64,64,256]),
            Identity_block(256,[64,64,256])
        )
        self.model3 = nn.Sequential(
            Conv_block(256,[128,128,512]),
            Identity_block(512,[128,128,512]),
            Identity_block(512,[128,128,512]),
            Identity_block(512,[128,128,512])
        )
        
        self.conv1 = Conv_block(512,[256,256,1024])
        self.loop_identity = Identity_block(1024,[256,256,1024])
        self.model4 = nn.Sequential(
            Conv_block(1024,[512,512,2048]),
            Identity_block(2048,[512,512,2048]),
            Identity_block(2048,[512,512,2048])
        )
        
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.model5 = nn.Sequential(
            nn.Linear(2048,1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024,num_classes)
        )
        def forward(self,x):
            c1 = x = self.model1(x)
            c2 = x = self.model2(x)
            c3 = x = self.model3(x)
            x = self.conv1(x)
            for i in range(22):
                x = self.loop_identity(x)
            c4 = x
            c5 = x = self.model4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0),-1)
            x = self.model5(x)
            return c1,c2,c3,c4,c5,x
```
针对重新构建的代码，读者可自行对照前文的代码，感受不同框架之间的差异和优缺点。