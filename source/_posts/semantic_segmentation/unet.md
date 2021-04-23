---
title: U-net
date: 2021-04-23 10:48:06
top: true
cover: true
categories:
- Deep Learning
tags:
- semantic segmentation
- U-net
- Fanrencli
author: Fanrencli
---

## 深度学习之语义分割U-net(2015)

### what is U-net

上篇文章我们介绍了`SegNet`网络的相关知识和网络的特点，这篇文章介绍2015年发表的另一篇比较经典的语义分割网络`U-net`，这篇网络与上篇的`SegNet`的网络差别不大，但是结构上还是有差别的，并且在`U-net`网络中运用了跳跃连接，这个特点和`SegNet`不同，也正是因为此处的不同，有的文章将这两个网络的池化索引和跳跃连接这两个特点进行结合进行网络构建。

`U-net`网络结构与`SegNet`结构相似，都拥有编码和解码的过程，但是在有效特征层选取的时候，`U-net`选取了多个特征层进行特征融合，而`SegNet`网络值利用了最后一层特征。

![U-net网络结构](http://39.105.26.229:4567/20191109101342389.png)

### 主干网络结构简介

针对`U-net`网络，我们选取`MobileNet`作为主干网络，在之前的文章中已经介绍过，且给出了相关代码，此处给出[链接](http://fanrencli.cn/2021/04/20/feature-extractor/mobilenet/)。

### 特征解码

特征解码过程与`SegNet`网络类似，针对编码过程中提取得到的有效特征层进行上采样解码，并与对应的有效特征层进行连接。

代码如下：
```python

from keras.models import *
from keras.layers import *
from nets.mobilenet import get_mobilenet_encoder
MERGE_AXIS = -1

def _unet( n_classes , encoder , l1_skip_conn=True,  input_height=416, input_width=608  ):

	img_input , levels = encoder( input_height=input_height ,  input_width=input_width )
	[f1 , f2 , f3 , f4 , f5 ] = levels 

	o = f4
	# 26,26,512
	o = ZeroPadding2D((1,1))(o)
	o = Conv2D(512, (3, 3), padding='valid')(o)
	o = BatchNormalization()(o)

	# 52,52,512
	o = ( UpSampling2D((2,2)))(o)
	# 52,52,768
	o =  concatenate([o, f3],axis=MERGE_AXIS )  
	o = ZeroPadding2D((1,1))(o)
	# 52,52,256
	o = Conv2D(256, (3, 3), padding='valid')(o)
	o = BatchNormalization()(o)

	# 104,104,256
	o = UpSampling2D( (2,2)))(o)
	# 104,104,384
	o = concatenate([o,f2],axis=MERGE_AXIS ) )
	o = ZeroPadding2D((1,1))(o)
	# 104,104,128
	o = Conv2D(128 , (3, 3), padding='valid')(o)
	o = BatchNormalization()(o)
	# 208,208,128
	o = UpSampling2D((2,2))(o)
	
	if l1_skip_conn:
		o = concatenate([o,f1],axis=MERGE_AXIS )

	o = ZeroPadding2D((1,1))(o)
	o = Conv2D( 64 , (3, 3), padding='valid')(o)
	o = BatchNormalization()(o)

	o =  Conv2D(n_classes, (3, 3), padding='same')( o )
	
	# 将结果进行reshape
	o = Reshape((int(input_height/2)*int(input_width/2), -1))(o)
	o = Softmax()(o)
	model = Model(img_input,o)
	return model

def mobilenet_unet( n_classes ,  input_height=224, input_width=224 , encoder_level=3):
	model =  _unet( n_classes , get_mobilenet_encoder ,  input_height=input_height, input_width=input_width  )
	return model
```

至此，`U-net`相关的代码就介绍完成了，在后期会针对已经介绍的网络进行不断更新。


