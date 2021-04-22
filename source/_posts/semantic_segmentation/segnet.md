---
title: SegNet
date: 2021-04-22 14:45:35
top: true
cover: true
categories:
- Deep Learning
tags:
- semantic segmentation
- SegNet
- Fanrencli
author: Fanrencli
---

## 深度学习之语义分割SegNet

### what is SegNet?

Emmmmm，从这篇文章开始应该会集中更新语义分割系列的算法模型，主要是科研的需求吧，应该会针对所有比较流行的语义分割的算法进行总结，以及讲解，针对各种算法的特点进行体现，而其中利用的主干特征提取的网络会在之前讲解过的特征提取网络中选择。总的来说，接下来一段时间会集中更新语义分割的相关系列算法，并对其中比较优秀的算法进行代码方面的实现。So，接下来就开始我们的第一次更新的内容吧——SegNet。

想要了解`SegNet`，那我们首先要对语义分割这个概念进行了解。所谓的语义分割其实就是针对图像中的像素进行分类，判断每个像素所属的类别，这就是我们需要做的任务，其实从根本上来看，就是像素的多分类问题。

![图1](http://39.105.26.229:4567/v2-c2bb4d18413382fccbc7455a1def8af6_1440w.jpg)

语义分割在实际生活中的应用主要体现在医学影像和无人驾驶方面，在医学影像中主要用于对一些图像中的异常（如肿瘤）进行分割，辅助医生对病人的病情进行判断。在无人驾驶中，语义分割的作用就体现在对道路、车辆、行人、道路标识等一些物体的判断，如果语义分割能够对这些方面做得很好，那么对于社会的发展将是一个重要的推动力。

而`SegNet`网络就是语义分割算法的一种，其网络结构符合语义分割算法的基本结构特征，主要包括编码和解码两个过程，根据最后的解码结果与真实值之间的比较训练模型参数。主要过程就是先通过特征提取进行降维，再进行上采样升维得到一个具有一定会长宽的特层。

![SegNet网络结构](http://39.105.26.229:4567/20191108195650187.png)

### 主干网络结构简介

针对`SegNet`算法，其网络结构主要包括编码和解码两个部分。编码部分即特征提取部分，选择`ResNet50`作为特征提取的网络。在之前的文章中我们已经介绍过了`ResNet101`，相比于`ResNet101`的网络结构，`ResNet50`只在其中的`Identity`基础块的重复次数降低了，其余部分并没有过多的变化，其网络结构如下图：

![ResNet50网络结构](http://39.105.26.229:4567/20191028224145980.png)

`ResNet50`网络的主要流程：
- 由于第一层7x7卷积，所以在卷积前加入`ZeroPadding2D`，然后如同图中所示，归一化接一层激活和池化
- 第二层一次`Conv_block`、两次`identity_block`
- 第三层一次`Conv_block`、三次`identity_block`
- 第四层一次`Conv_block`、五次`identity_block`
- 第五层一次`Conv_block`、两次`identity_block`

代码如下：
```python
def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 1x1压缩
    x = Conv2D(filters1, (1, 1) , name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 3x3提取特征
    x = Conv2D(filters2, kernel_size , data_format=IMAGE_ORDERING , padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 1x1扩张特征
    x = Conv2D(filters3 , (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # 残差网络
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

# 与identity_block最大差距为，其可以减少wh，进行压缩
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 1x1压缩
    x = Conv2D(filters1, (1, 1) , strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 3x3提取特征
    x = Conv2D(filters2, kernel_size , padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 1x1扩张特征
    x = Conv2D(filters3, (1, 1) , name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # 1x1扩张特征
    shortcut = Conv2D(filters3, (1, 1) , strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    # add
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
def get_resnet50_encoder(input_height=224 , input_width=224):

    assert input_height%32 == 0
    assert input_width%32 == 0

    img_input = Input(shape=(input_height,input_width , 3 ))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    # f1是hw方向压缩一次的结果
    f1 = x
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3) , strides=(2, 2))(x)
    
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # f2是hw方向压缩两次的结果
    f2 = x


    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # f3是hw方向压缩三次的结果
    f3 = x 

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    # f4是hw方向压缩四次的结果
    f4 = x 

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    # f5是hw方向压缩五次的结果
    f5 = x 

    x = AveragePooling2D((7, 7) , name='avg_pool')(x)
    return img_input , [f1 , f2 , f3 , f4 , f5  ]
```

### 特征解码

在特征解码部分，主要是针对前面编码部分提取出来的特征进行解码，然后为每个像素进行分类，获取loss值进行训练。其中解码部分主要就是对应于之前提取到的特征进行上采样，将特征纬度不断扩大。

代码如下：
```python
def segnet_decoder( f , n_classes, n_up=3 ):

	assert n_up >= 2

	o = f
	o = ZeroPadding2D((1,1))(o)
	o = Conv2D(512, (3, 3), padding='valid')(o)
	o = BatchNormalization()(o)
	# 进行一次UpSampling2D，此时hw变为原来的1/8
	o = UpSampling2D((2,2))(o)
	o = ZeroPadding2D((1,1))(o)
	o = Conv2D( 256, (3, 3), padding='valid')(o)
	o = BatchNormalization()(o)

	# 进行一次UpSampling2D，此时hw变为原来的1/4
	for _ in range(n_up-2):
		o = UpSampling2D((2,2))(o)
		o = ZeroPadding2D((1,1))(o)
		o = Conv2D( 128 , (3, 3), padding='valid')(o)
		o = BatchNormalization()(o)

	# 进行一次UpSampling2D，此时hw变为原来的1/2
	o = UpSampling2D((2,2) )(o)
	o = ZeroPadding2D((1,1) )(o)
	o = Conv2D( 64 , (3, 3), padding='valid' )(o)
	o = BatchNormalization()(o)

	# 此时输出为h_input/2,w_input/2,nclasses
	o = Conv2D(n_classes , (3, 3), padding='same')( o )
	return o 
def _segnet(n_classes , encoder, input_height=416, input_width=416 , encoder_level=3):
	# encoder通过主干网络
	img_input , levels = encoder( input_height=input_height ,  input_width=input_width )
	# 获取hw压缩四次后的结果
	feat = levels[encoder_level]
	# 将特征传入segnet网络
	o = segnet_decoder(feat, n_classes, n_up=3 )
	# 将结果进行reshape
	o = Reshape((int(input_height/2)*int(input_width/2), -1))(o)
	o = Softmax()(o)
	model = Model(img_input,o)
	return model

def resnet50_segnet( n_classes ,  input_height=416, input_width=416 , encoder_level=3):

	model = _segnet( n_classes , get_resnet50_encoder ,  input_height=input_height, input_width=input_width , encoder_level=encoder_level)
	return model
```
根据以上的代码，我们很容易看出，解码部分对应于编码的部分，通过不断的上采样将特征放大到原来图片的1/2的大小然后输出分类。
至此，关于`SegNet`代码的所有内容就基本讲完了，下一张会讲解医学影像处理常用的语义分割网络`U-net`。

