---
title: VGG
date: 2021-04-18 14:06:34
categories:
- Deep Learning
tags:
- VGG16
- Fanrencli
author: Fanrencli
---
## 深度学习神经网络特征提取（二）

### 网络结构

要构建`VGG16`特征提取网络，我们只需要了解：

- VGG16网络层数：VGG16,VGG19

在本文中，根据实际的项目要求，构建了VGG16网络，相对于ResNet网络较为简单，整个网络只是一个线性的结构，没有分支，所以在VGG的学习中只需要掌握基础的深度学习原理就可以构建出来VGG网络。
![VGG网络结构](http://39.106.34.39:4567/v2-ea924e733676e0da534f677a97c98653_720w.jpg)
![VGG16网络结构](http://39.106.34.39:4567/2019101614532169.png)

### 构建VGG16网络结构

VGG16的网络有三种不同的层构成，分别是卷积、池化和全连接，具体的运行方式：
- 输入一张尺寸为（224，224，3）的图片
- 两次(3,3)卷积，层数64，输出为(224,224,64)，(2，2)最大池化，输出(112,112,64)。
- 两次(3,3)卷积，层数128，输出为(112,112,128)，(2，2)最大池化，输出(56,56,64)。
- 三次(3,3)卷积，层数256，输出为(56,56,64)，(2，2)最大池化，输出(28,28,64)。
- 三次(3,3)卷积，层数512，输出为(28,28,64)，(2，2)最大池化，输出(14,14,64)。
- 三次(3,3)卷积，层数512，输出为(14,14,64)，(2，2)最大池化，输出(7,7,64)。
- 对结果进行平铺，接上两层4096的全连接层
- 最后全连接进行分类

代码如下：
```python
    def get_VGG16_model(input_shape,classes):
        image_input = Input(shape = input_shape)

        x = layers.Conv2D(64,(3,3),padding = 'same', activation = 'relu')(image_input)
        x = layers.Conv2D(64,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.MaxPooling2D((2,2),strides =(2,2))(x)

        x = layers.Conv2D(128,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.Conv2D(128,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.MaxPooling2D((2,2),strides =(2,2))(x)

        x = layers.Conv2D(256,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.Conv2D(256,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.Conv2D(256,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.MaxPooling2D((2,2),strides =(2,2))(x)

        x = layers.Conv2D(512,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.Conv2D(512,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.Conv2D(512,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.MaxPooling2D((2,2),strides =(2,2))(x)

        x = layers.Conv2D(512,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.Conv2D(512,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.Conv2D(512,(3,3),padding = 'same', activation = 'relu')(x)
        x = layers.MaxPooling2D((2,2),strides =(2,2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(4096,activation = 'relu')(x)
        x = layers.Dense(4096,activation = 'relu')(x)
        x = layers.Dense(classes, activation = 'softmax')(x) 
        model = models.Model(image_input,x)
        return model
```

最后，根据VGG16的网络结构，大家可以自行尝试构建VGG19的网络结构。