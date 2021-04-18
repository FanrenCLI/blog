---
title: ResNet
date: 2021-04-18 13:19:58
top: true
cover: true
categories:
    - Deep Learning
tags:
    - ResNet101
    - Fanrencli
author: Fanrencli
---
## 深度学习神经网络特征提取（一）

### 网络结构

要构建`ResNet`特征提取网络，我们只需要了解两个方面：

- ResNet网络层数：Resnet50,ResNet101,ResNet152
- ResNet网络基础组成：Conv_Block,Identity_Block

在本文中，根据实际的项目要求，构建了RseNet101网络。
![ResNet网络结构](http://39.105.26.229:4567/20180114205444652.png)
![ResNet基础Backbone](http://39.105.26.229:4567/20180114184946861.png)

### 构建Conv_Block模块
针对`Conv_Block`模块，我们首先要了解这个模块的具体结构，`Conv_Block`从输入开始分两支分别进行特征提取，以一次卷积、一次归一化、一次`ReLu`激活函数的形式连接三次，并在第一次的卷积层步长为2进行降维，另一分支只进行一次步长为2的卷积、一次归一化，然后将两个分支进行连接，再一次激活函数层完成一次`Conv_Block`，结构如下图：
![Conv_Block](http://39.105.26.229:4567/20191113094201415.png)
```python
    def Conv_block(input_feature,kernel_size,filters,strides = (2,2)):
        filter1,filter2,filter3 = filters
        #first line conv
        x = layers.Conv2D(filter1,(1,1),strides = strides, use_bias=True)(input_feature)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filter2,(kernel_size,kernel_size),padding = 'same', use_bias=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filter3,(1,1), use_bias=True)(x)
        x = layers.BatchNormalization()(x)

        #parallel conv
        shortcut = layers.Conv2D(filter3,(1,1), strides = strides, use_bias=True)(input_feature)
        shortcut = layers.BatchNormalization()(shortcut)

        # add the parallel conv

        x = layers.Add()([x,shortcut])
        x = layers.Activation('relu')(x)
        return x
```

### 构建Identity_Block模块

`Identity_Block`模块不同于`Conv_Block`模块，`Identity_Block`模块只对特征进行提取，即只进行深度的堆叠不行进降维，所以在结构上与`Conv_Block`相似——同样是双分支结构，在另一分支上不进行操作，只是将输入与另一分支的结果进行叠加，具体结构如下：

![Identity_Block](http://39.105.26.229:4567/20191113094135752.png)

代码如下：
```python
    def identity_block(input_feature,kernel_size,filters):
        filter1,filter2,filter3 = filters

        x = layers.Conv2D(filter1,(1,1), use_bias=True)(input_feature)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filter2,(kernel_size,kernel_size), padding = 'same', use_bias=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filter3,(1,1), use_bias=True)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([x,input_feature])
        x = layers.Activation('relu')(x)
        return x
```

### 构建ResNet101网络

构建ResNet101网络的具体形式如上图所示。
- 由于第一层7x7卷积，所以在卷积前加入`ZeroPadding2D`，然后如同图中所示，归一化接一层激活和池化
- 第二层一次`Conv_block`、两次`identity_block`
- 第三层一次`Conv_block`、三次`identity_block`
- 第四层一次`Conv_block`、22次`identity_block`
- 第五层一次`Conv_block`、两次`identity_block`

在第五层之后本文接上一层全局池化以及两层全连接进行分类，但是ResNset网络在第五层基本就结束了，后续根据个人的需求进行修改即可，本文只是做了一个图片的分类的实例，接上后续的全连接分类。
代码如下：
```python
    def get_resnet_model(input_shape,classes):
        input_image = Input(shape = input_shape)
        x = layers.ZeroPadding2D((3,3))(input_image)
        x = layers.Conv2D(64,(7,7),strides=(2,2),  use_bias=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        c1 = x = layers.MaxPooling2D((3,3),strides=(2, 2), padding='same')(x)

        x = Conv_block(x, 3, [64,64,256], strides=(1,1))
        x = identity_block(x,3,[64,64,256])
        c2 = x = identity_block(x,3,[64,64,256])
        x = Conv_block(x,3,[128,128,512])
        x = identity_block(x,3,[128,128,512])
        x = identity_block(x,3,[128,128,512])
        c3 = x =identity_block(x,3,[128,128,512])
        x = Conv_block(x,3,[256,256,1024])
        for i in range(22):
            x = identity_block(x,3,[256,256,1024])
        c4 = x
        x = Conv_block(x,3,[512,512,2048])
        x = identity_block(x,3,[512,512,2048])
        c5 = x = identity_block(x,3,[512,512,2048])
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024,activation = 'relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(classes,activation = 'softmax')(x)
        model = models.Model(input_image,output)
        return model
```
