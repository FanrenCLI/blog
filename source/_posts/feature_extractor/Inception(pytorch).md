---
title: Inception(pytorch)
date: 2021-05-09 18:38:08
categories: 
- Deep Learning
tags:
- Inceptionv3
- Xception
- Pytorch
author: Fanrencli
---

## 深度学习神经网络特征提取（七）

emmmmmmm....代码重构的过程实在是无聊与枯燥，烦躁的心情也无法抑制，不过运动可以缓解一下。整天坐在实验室，整个人就是个关禁闭的状态啊~~~感觉近期就是处于一个懵逼的状态:-( 
哎，不说了，开始今天的文章吧。。。本次文章给出`Inception`系列的`Pytorch`版本的代码。关于网络的讲解部分，大家参考前期的[文章](http://fanrencli.cn/2021/04/20/feature-extractor/inception/)


### Inceptionv3

```python
def conv2d_bn(input_channel, output_channel, kernel_size, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(input_channel,output_channel,kernel_size=kernel_size,padding=padding,stride=stride,bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace = True)
    )
```
```python
class Inceptionv3_frist_block(nn.Module):
    def __init__(self,input_channel,change_channel):
        super(Inceptionv3_frist_block,self).__init__()
        self.first_branch = conv2d_bn(input_channel=input_channel,output_channel=64,kernel_size=1,stride=1)
        self.second_branch = nn.Sequential(
            conv2d_bn(input_channel=input_channel,output_channel=48,kernel_size=1),
            conv2d_bn(input_channel=48,output_channel=64,kernel_size=5,padding=2)
        )
        self.third_branch = nn.Sequential(
            conv2d_bn(input_channel=input_channel,output_channel=64,kernel_size=1),
            conv2d_bn(input_channel=64,output_channel=96,kernel_size=3,padding=1),
            conv2d_bn(input_channel=96,output_channel=96,kernel_size=3,padding=1)
        )
        self.forth_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            conv2d_bn(input_channel=input_channel,output_channel=change_channel,kernel_size=1)
        )
    def forward(self,x):
        branch1 = self.first_branch(x)
        branch2 = self.second_branch(x)
        branch3 = self.third_branch(x)
        branch4 = self.forth_branch(x)
        x = torch.cat((branch1,branch2,branch3,branch4),1)
        return x
```
```python
class Inceptionv3_second_block(nn.Module):
    def __init__(self,input_channel,change_channel=0,is_first_block_part=False):
        super(Inceptionv3_second_block,self).__init__()
        self.is_first_block_part = is_first_block_part
        if is_first_block_part:
            self.first_branch = conv2d_bn(input_channel=input_channel,output_channel=384,kernel_size=3,stride=2)
            self.second_branch = nn.Sequential(
                conv2d_bn(input_channel=input_channel,output_channel=64,kernel_size=1),
                conv2d_bn(input_channel=64,output_channel=96,kernel_size=3,padding=1),
                conv2d_bn(input_channel=96,output_channel=96,kernel_size=3,stride=2)
            )
            self.third_branch = nn.MaxPool2d(kernel_size=3,stride=2)
        else:
            self.first_branch = conv2d_bn(input_channel=input_channel,output_channel=192,kernel_size=1)
            self.second_branch = nn.Sequential(
                conv2d_bn(input_channel=input_channel,output_channel=change_channel,kernel_size=1),
                conv2d_bn(input_channel=change_channel,output_channel=change_channel,kernel_size=[1,7],padding=[0,3]),
                conv2d_bn(input_channel=change_channel,output_channel=192,kernel_size=[7,1],padding=[3,0])
            )
            self.third_branch = nn.Sequential(
                conv2d_bn(input_channel=input_channel,output_channel=change_channel,kernel_size=1),
                conv2d_bn(input_channel=change_channel,output_channel=change_channel,kernel_size=[7,1],padding=[3,0]),
                conv2d_bn(input_channel=change_channel,output_channel=change_channel,kernel_size=[1,7],padding=[0,3]),
                conv2d_bn(input_channel=change_channel,output_channel=change_channel,kernel_size=[7,1],padding=[3,0]),
                conv2d_bn(input_channel=change_channel,output_channel=192,kernel_size=[1,7],padding=[0,3])
            )
            self.forth_branch = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride = 1, padding=1),
                conv2d_bn(input_channel=input_channel,output_channel=192,kernel_size=1)
            )
    def forward(self,x):
        branch1 = self.first_branch(x)
        branch2 = self.second_branch(x)
        branch3 = self.third_branch(x)
        if self.is_first_block_part:
            x = torch.cat((branch1,branch2,branch3),1)
        else:
            branch4 = self.forth_branch(x)
            x = torch.cat((branch1,branch2,branch3,branch4),1)
        return x
```
```python
class Inceptionv3_third_block(nn.Module):
    def __init__(self,input_channel,is_first_block_part=False):
        super(Inceptionv3_third_block,self).__init__()
        self.is_first_block_part = is_first_block_part
        if is_first_block_part:
            self.first_branch = nn.Sequential(
                conv2d_bn(input_channel=input_channel,output_channel=192,kernel_size=1),
                conv2d_bn(input_channel=192,output_channel=320,kernel_size=3,stride=2)
            )
            self.second_branch = nn.Sequential(
                conv2d_bn(input_channel=input_channel,output_channel=192,kernel_size=1),
                conv2d_bn(input_channel=192,output_channel=192,kernel_size=[1,7],padding=[0,3]),
                conv2d_bn(input_channel=192,output_channel=192,kernel_size=[7,1],padding=[3,0]),
                conv2d_bn(input_channel=192,output_channel=192,kernel_size=3,stride=2)
            )
            self.third_branch = nn.MaxPool2d(kernel_size=3,stride=2)
        else:
            #branch1x1
            self.conv1 = conv2d_bn(input_channel=input_channel,output_channel=320,kernel_size=1)
            #branch3x3
            self.conv2 = conv2d_bn(input_channel=input_channel,output_channel=384,kernel_size=1)
            #branch3x3_1
            #branch3x3_2
            self.conv3 = conv2d_bn(input_channel=384,output_channel=384,kernel_size=[1,3],padding=[0,1])
            self.conv4 = conv2d_bn(input_channel=384,output_channel=384,kernel_size=[3,1],padding=[1,0])
            #branch3x3db1
            self.conv5 = conv2d_bn(input_channel=input_channel,output_channel=448,kernel_size=1)
            self.conv6 = conv2d_bn(input_channel=448,output_channel=384,kernel_size=3,padding=1)
            #branch3x3db1_1
            #branch3x3db1_2
            self.conv7 = conv2d_bn(input_channel=384,output_channel=384,kernel_size=[1,3],padding=[0,1])
            self.conv8 = conv2d_bn(input_channel=384,output_channel=384,kernel_size=[3,1],padding=[1,0])

            self.avg = nn.AvgPool2d(kernel_size=3,stride=1,padding =1)
            self.conv9 = conv2d_bn(input_channel=input_channel,output_channel=192,kernel_size=1)
            
    def forward(self,x):
        if self.is_first_block_part:
            branch1 = self.first_branch(x)
            branch2 = self.second_branch(x)
            branch3 = self.third_branch(x)
            x = torch.cat((branch1,branch2,branch3),1)
        else:
            branch1 = self.conv1(x)

            branch2 = self.conv2(x)
            branch2_1 = self.conv3(branch2)
            branch2_2 = self.conv4(branch2)
            branch2 = torch.cat((branch2_1,branch2_2),1)

            branch3 = self.conv5(x)
            branch3 = self.conv6(branch3)
            branch3_1 = self.conv7(branch3)
            branch3_2 = self.conv8(branch3)
            branch3 = torch.cat((branch3_1,branch3_2),1)

            branch4 = self.avg(x)
            branch4 = self.conv9(branch4)
            x = torch.cat((branch1,branch2,branch3,branch4),1)
        return x
```
```python
class Inceptionv3(nn.Module):
    def __init__(self,num_classes):
        super(Inceptionv3,self).__init__()
        # input_shape  = 3,299,299
        self.model1 = nn.Sequential(
            # 299,299 -> 149,149 
            conv2d_bn(input_channel=3,output_channel=32,kernel_size=3,stride=2),
            # 149,149 -> 147,147
            conv2d_bn(input_channel=32,output_channel=32,kernel_size=3),
            # 147,147 -> 147,147
            conv2d_bn(input_channel=32,output_channel=64,kernel_size=3,padding=1),
            # 147,147 -> 73,73
            nn.MaxPool2d(kernel_size=3,stride=2),
            # 73,73 -> 73,73
            conv2d_bn(input_channel=64,output_channel=80,kernel_size=1),
            # 73,73 -> 71,71
            conv2d_bn(input_channel=80,output_channel=192,kernel_size=3),
            # 71,71 -> 35,35
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        # 35,35,192 -> 35,35,288
        self.block1 = nn.Sequential(
            Inceptionv3_frist_block(input_channel = 192,change_channel = 32),
            Inceptionv3_frist_block(input_channel = 256,change_channel = 64),
            Inceptionv3_frist_block(input_channel = 288,change_channel = 64)
        )
        #35,35,288 -> 17,17,768
        self.block2 = nn.Sequential(
            Inceptionv3_second_block(input_channel=288,is_first_block_part=True),
            Inceptionv3_second_block(input_channel=768,change_channel=128),
            Inceptionv3_second_block(input_channel=768,change_channel=160),
            Inceptionv3_second_block(input_channel=768,change_channel=160),
            Inceptionv3_second_block(input_channel=768,change_channel=192)
        )
        #17,17,768 -> 8,8,2048
        self.block3 = nn.Sequential(
            Inceptionv3_third_block(768,is_first_block_part=True),
            Inceptionv3_third_block(1280),
            Inceptionv3_third_block(2048),
        )
        self.avg = nn.AdaptiveMaxPool2d(1)
        self.linear = nn.Linear(2048,num_classes)
    def forward(self,x):
        x = self.model1(x)
        x =self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avg(x)
        x = x.view(x.size(0),x.size(1))
        x = self.linear(x)
        return x
```
emmmmmmm...????重新写完一个竟然用了好几个小时？？？？？看来是人老了:-(

### Xception

```python 
class Xception_Entry_flow(nn.Module):
    def __init__(self,input_channel,change_channel):
        super(Xception_Entry_flow,self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channel,change_channel,kernel_size=1,stride=2),
            nn.BatchNorm2d(change_channel)
        )
        self.model = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel,input_channel,kernel_size=3,stride = 1,padding =1,groups =input_channel,bias=False),
            nn.Conv2d(input_channel,change_channel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(change_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(change_channel,change_channel,kernel_size=3,stride = 1,padding =1,groups =change_channel,bias=False),
            nn.Conv2d(change_channel,change_channel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(change_channel),
            nn.MaxPool2d(kernel_size=3,stride= 2,padding=1)
        )
    def forward(self,x):
        residual = self.shortcut(x)
        x = self.model(x)
        x += residual
        return x
```

```python 
class Xception_Middle_flow(nn.Module):
    def __init__(self,input_channel):
        super(Xception_Middle_flow,self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel,input_channel,kernel_size=3,stride = 1,padding =1,groups =input_channel,bias=False),
            nn.Conv2d(input_channel,input_channel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel,input_channel,kernel_size=3,stride = 1,padding =1,groups =input_channel,bias=False),
            nn.Conv2d(input_channel,input_channel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel,input_channel,kernel_size=3,stride = 1,padding =1,groups =input_channel,bias=False),
            nn.Conv2d(input_channel,input_channel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(input_channel),
        )
    def forward(self,x):
        shortcut = x
        x = self.model(x)
        x += shortcut
        return x
```

```python 
class Xception_Exit_flow(nn.Module):
    def __init__(self,input_channel):
        super(Xception_Exit_flow,self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channel,1024,kernel_size=1,stride=2,bias=False),
            nn.BatchNorm2d(1024)
        )
        self.model1 = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Conv2d(input_channel,input_channel,kernel_size=3,stride = 1,padding =1,groups =input_channel,bias=False),
            nn.Conv2d(input_channel,input_channel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(input_channel,input_channel,kernel_size=3,stride = 1,padding =1,groups =input_channel,bias=False),
            nn.Conv2d(input_channel,1024,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size=3,stride = 1,padding =1,groups =1024,bias=False),
            nn.Conv2d(1024,1536,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace = True),
            nn.Conv2d(1536,1536,kernel_size=3,stride = 1,padding =1,groups =1536,bias=False),
            nn.Conv2d(1536,2048,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace = True),
            nn.AdaptiveMaxPool2d(1)
        )
    def forward(self,x):
        residual = self.shortcut(x)
        x = self.model1(x)
        x +=residual
        x = self.model2(x)
        return x
```

```python 
class Xception(nn.Module):
    def __init__(self,num_classes):
        super(Xception,self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=2,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.Entry_flow = nn.Sequential(
            Xception_Entry_flow(64,128),
            Xception_Entry_flow(128,256),
            Xception_Entry_flow(256,728)
        )
        self.Middle_flow = nn.Sequential(
            Xception_Middle_flow(728),
            Xception_Middle_flow(728),
            Xception_Middle_flow(728),
            Xception_Middle_flow(728),
            Xception_Middle_flow(728),
            Xception_Middle_flow(728),
            Xception_Middle_flow(728),
            Xception_Middle_flow(728)
        )
        self.Exit_flow = Xception_Exit_flow(728)
        self.linear = nn.Linear(2048,num_classes)
    def forward(self,x):
        x = self.first_block(x)
        x = self.Entry_flow(x)
        x = self.Middle_flow(x)
        x = self.Exit_flow(x)
        x = x.view(x.size(0),x.size(1))
        x = self.linear(x)
        return x
```

至此，关于之前所有用`keras`构建的特征提取网络已经全部重构完毕，后续看情况可能会继续更新一下`DenseNet`、`ShuffleNet`和其他组合型特征提取网络。emmmmmmm....话说又用了一个上午:-(
对了，如果有读者想要看一下网络结构的细节部分，此处提供一下代码，此处代码可以进行适当修改，无缝链接到其他文章中的网络。
```python
net = Xception(10)
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