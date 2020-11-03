---
title: Haar Wavelet（1）
cover: true
top: true
categories:
  - 小波变换
tags:
  - Haar Wavelet
  - python
author: Fanrencli
---
>haar小波变换的图像应用实例

## Haar Wavelet 图片示例
![Picture resolutions(512*512)](http://39.105.26.229:4567/pic.png)
```python
import numpy as np
import cv2
import math
import requests

# numpy数组归一化
def Normalize(img):
    _range = np.max(img) - np.min(img)
    return (img - np.min(img)) / _range

#inputdata: row or column of image
#outputdata: array handle with haar_wavelet
def haar_dwt(row_or_col):#图片需要为2的次方形状
    length=len(row_or_col)//2
    Low_frequency=np.zeros(length,dtype=float)
    High_frequency=np.zeros(length,dtype=float)
    #小波变换的主体部分
    for i in range(length):
        Low_frequency[i]=(row_or_col[2*i]+row_or_col[2*i+1])/math.sqrt(2)
        High_frequency[i]=(row_or_col[2*i]-row_or_col[2*i+1])/math.sqrt(2)
    return np.append(Low_frequency,High_frequency)

#inputdata: array of image
#outputdata: array of image handel with haar_wavelet
def haar_dwt2D(img):
    col_num=img.shape[1]
    row_num=img.shape[0]
    half_col_num=col_num//2
    half_row_num=row_num//2
    for i in range(row_num):
        img[i]=haar_dwt(img[i])
    for j in range(col_num):
        img[:,j]=haar_dwt(img[:,j])
    #为了强化图片的显示效果，对数据进行归一化处理
    img[0:half_row_num,0:half_col_num]=Normalize(img[0:half_row_num,0:half_col_num])
    img[half_row_num:row_num,0:half_col_num]=Normalize(img[half_row_num:row_num,0:half_col_num])
    img[0:half_row_num,half_col_num:col_num]=Normalize(img[0:half_row_num,half_col_num:col_num])
    img[half_row_num:row_num,half_col_num:col_num]=Normalize(img[half_row_num:row_num,half_col_num:col_num])
    return img
if __name__ == '__main__':
    #读取网络图片（2选1）
    file_pic=requests.get('http://39.105.26.229:4567/pic.png')
    img= cv2.imdecode(np.fromstring(file_pic.content, np.uint8), 0).astype(np.float64)
    #读取本地图片（2选1）
    # img= cv2.imread("pic.png",0).astype(np.float64)
    cv2.imshow('asd',haar_dwt2D(img))
    cv2.waitKey(0)
    for i in range(0,img.shape[0]//2-1,img.shape[0]//2-1):
        for j in range(0,img.shape[1]//2-1,img.shape[1]//2-1):
            img[i:i+img.shape[0]//2,j:j+img.shape[1]//2]=haar_dwt2D(img[i:i+img.shape[0]//2,j:j+img.shape[1]//2])
    cv2.imshow('asd',img)
    cv2.waitKey(0)  
```