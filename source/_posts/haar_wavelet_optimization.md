---
title: Haar Wavelet（2）
cover: true
top: true
categories:
  - 小波变换
tags:
  - Haar Wavelet
  - python
author: Fanrencli
---
>haar小波变换的图像应用实例(优化)

## 优化目标
- 针对一维变换的循环处理方式，将使用矩阵运算进行代替
- 针对二维的分行、列的处理方法，使用矩阵运算代替
- 边界处理

### 一维处理
观察[上篇文章](https://fanrencli.cn/2020/10/27/haar-wavelet/)的代码,其中的一维变换代码``，使用简单的循环处理方法，如下:
```python 
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
```
显然此种循环处理方法的性能不足，针对这个缺点，本文采用矩阵运算进行代替。首先分析循环的处理过程，转为矩阵方式，如下:
$$Matrix_{low\_frequency}=
{1\over \sqrt{2}}
\begin{pmatrix}
1&1&0&\cdots&\cdots&0\\
0&0&1&1&\cdots&0\\
\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\\
0&\cdots&\cdots&0&1&1\\
\end{pmatrix}（n*{n\over 2}）
$$
$$Matrix_{high\_frequency}=
{1\over \sqrt{2}}
\begin{pmatrix}
1&-1&0&\cdots&\cdots&0\\
0&0&1&-1&\cdots&0\\
\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\\
0&\cdots&\cdots&0&1&-1\\
\end{pmatrix}（n*{n\over 2}）
$$
$$
Low\_frequency=Matrix_{low\_frequency}
\begin{pmatrix}
x_1\\
x_2\\
\vdots\\
x_n\\
\end{pmatrix}
$$
$$
High\_frequency=Matrix_{high\_frequency}
\begin{pmatrix}
x_1\\
x_2\\
\vdots\\
x_n\\
\end{pmatrix}
$$
$$
img_{row\_or\_col}=
\begin{pmatrix}
Low\_frequency\\
High\_frequency\\
\end{pmatrix}
$$
由此分析，针对代码进行优化，将矩阵构造处理抽取出来作为一个函数`Create_haar_matrix`，这样便于以后根据图像分辨率进行构建矩阵，并且将$Matrix_{low\_frequency}$与$Matrix_{high\_frequency}$合成为一个矩阵T,有以下的运算过程:
$$
T=
\begin{pmatrix}
Matrix_{low\_frequency}\\
Matrix_{high\_frequency}\\
\end{pmatrix}
$$
$$
img_{row\_or\_col}=T
\begin{pmatrix}
x_1\\
x_2\\
\vdots\\
x_n\\
\end{pmatrix}
$$
代码如下:
```python
#根据输入数组长度创建小波变换矩阵
def Create_haar_matrix(length):
    half_length=length//2
    haar_wavelet_matrix=np.zeros((length,length),dtype=float)
    for i in range(half_length):
        haar_wavelet_matrix[i,i*2:i*2+1]=1/math.sqrt(2)
        haar_wavelet_matrix[half_length+i,i*2]=1/math.sqrt(2)
        haar_wavelet_matrix[half_length+i,i*2+1]=-1/math.sqrt(2)
    return haar_wavelet_matrix

#inputdata: row or column of image
#outputdata: array handle with haar_wavelet
def haar_dwt(row_or_col,haar_wavelet_matrix):
    Low_High_frequency=np.dot(haar_wavelet_matrix,row_or_col)
    return Low_High_frequency

```
### 二维处理
观察上篇文章的代码,其中的二维变换代码`haar_dwt2D`，使用简单的循环处理方法，如下:
```python
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
    #归一化处理
    img[0:half_row_num,0:half_col_num]=Normalize(img[0:half_row_num,0:half_col_num])
    img[half_row_num:row_num,0:half_col_num]=Normalize(img[half_row_num:row_num,0:half_col_num])
    img[0:half_row_num,half_col_num:col_num]=Normalize(img[0:half_row_num,half_col_num:col_num])
    img[half_row_num:row_num,half_col_num:col_num]=Normalize(img[half_row_num:row_num,half_col_num:col_num])
    return img
```
此处代码，分行列分别进行运算，由一维变换过程:
$$
img_{row\_or\_col}=T
\begin{pmatrix}
x_1\\
x_2\\
\vdots\\
x_n\\
\end{pmatrix}
$$
对此进行拓展得到：
$$
img=T\cdot
\begin{pmatrix}
x_{11}&x_{12}&\cdots&x_{1n}\\
x_{21}&\ddots&&\vdots\\
\vdots&&\ddots&\vdots\\
x_{n1}&\cdots&\cdots&x_{nn}\\
\end{pmatrix}
$$
进而代码如下：
```python
#inputdata: array of image
#outputdata: array of image handel with haar_wavelet
def haar_dwt2D(img):
    col_num=img.shape[1]
    row_num=img.shape[0]
    #创建小波变换矩阵
    haar_wavelet_matrix=Create_haar_matrix(col_num)
    half_col_num=col_num//2
    half_row_num=row_num//2
    img=haar_dwt(img,haar_wavelet_matrix)
    img=haar_dwt(img.T,haar_wavelet_matrix)
    img=img.T
    img[0:half_row_num,0:half_col_num]=Normalize(img[0:half_row_num,0:half_col_num])
    img[half_row_num:row_num,0:half_col_num]=Normalize(img[half_row_num:row_num,0:half_col_num])
    img[0:half_row_num,half_col_num:col_num]=Normalize(img[0:half_row_num,half_col_num:col_num])
    img[half_row_num:row_num,half_col_num:col_num]=Normalize(img[half_row_num:row_num,half_col_num:col_num])
    return img
```
紧接着对代码的这个部分进行思考：
```python
    img=haar_dwt(img,haar_wavelet_matrix)
    img=haar_dwt(img.T,haar_wavelet_matrix)
    img=img.T
```
此部分的代码逻辑过程可描述为：
$$
img_{pic}=
\begin{pmatrix}
x_{11}&x_{12}&\cdots&x_{1n}\\
x_{21}&\ddots&&\vdots\\
\vdots&&\ddots&\vdots\\
x_{n1}&\cdots&\cdots&x_{nn}\\
\end{pmatrix}
$$
$$
img_{row}=T \cdot img_{pic}
$$
$$
img_{col}=T \cdot (img_{row})^T
$$
$$
img_{pic}=(img_{col})^T
$$
即：
$$
img_{pic}= (T \cdot (T \cdot img_{pic})^T)^T=T\cdot img_{pic} \cdot T^T
$$
由此将代码进一步转化：
```python
#inputdata: row or column of image
#outputdata: array handle with haar_wavelet
def haar_dwt(img,haar_wavelet_matrix):
    Low_High_frequency=np.dot(np.dot(haar_wavelet_matrix,img),haar_wavelet_matrix.T)
    return Low_High_frequency

#inputdata: array of image
#outputdata: array of image handel with haar_wavelet
def haar_dwt2D(img):
    col_num=img.shape[1]
    row_num=img.shape[0]
    #创建小波变换矩阵
    haar_wavelet_matrix=Create_haar_matrix(col_num)
    half_col_num=col_num//2
    half_row_num=row_num//2
    img=haar_dwt(img,haar_wavelet_matrix)

    img[0:half_row_num,0:half_col_num]=Normalize(img[0:half_row_num,0:half_col_num])
    img[half_row_num:row_num,0:half_col_num]=Normalize(img[half_row_num:row_num,0:half_col_num])
    img[0:half_row_num,half_col_num:col_num]=Normalize(img[0:half_row_num,half_col_num:col_num])
    img[half_row_num:row_num,half_col_num:col_num]=Normalize(img[half_row_num:row_num,half_col_num:col_num])
    return img
```
### 边界处理
以上的处理基本解决了代码性能方面的不足，接着对代码的适用性进行考虑，此代码要求图片的分辨率必须为$2^x*2^x$，所以需要针对不是2的次方的图片进行扩展，思路：边界填充。
> 边缘填充:
> - 复制法：复制最边缘的像素
> - 反射法：对称轴
> - 外包装法：
> - 常量法：用常量值填充四周

本文选用其中的复制法，作为示例，在`python`中`OpenCV`提供了相关的方法`cv2.copyMakeBorder(src,top, bottom, left, right ,borderType,value)`。

- src : 需要填充的图像
- top : 图像上的填充边界长度
- bottom : 图像下面的填充边界长度
- left : 图像左边的填充边界长度
- right : 图像右边的填充边界长度
- borderType : 边界填充类型
- value : 填充边界的颜色，常用于常量法。

本文针对图像的分辨率，对于宽和高不是2的倍数，则对应边界添加一行复制行，本文代码用于每压缩一次则填充边界一次（即每次只填充一行，将行或列增加至2的倍数），并不是将边界一次扩展到2的次方倍，所以`haar_dwt2D`代码如下：
```python
def haar_dwt2D(img):
    col_num=img.shape[1]
    row_num=img.shape[0]
    if col_num%2!=0 and row_num%2!=0:
        img=cv2.copyMakeBorder(img,0,1,0,1,cv2.BORDER_REPLICATE)
    elif col_num%2==0 and row_num%2!=0:
        img=cv2.copyMakeBorder(img,0,1,0,0,cv2.BORDER_REPLICATE)
    elif col_num%2!=0 and row_num%2==0:
        img=cv2.copyMakeBorder(img,0,0,0,1,cv2.BORDER_REPLICATE)
    col_num=img.shape[1]
    row_num=img.shape[0]
    #创建小波变换矩阵
    haar_wavelet_matrix_row,haar_wavelet_matrix_col=Create_haar_matrix(row_num,col_num)
    half_col_num=col_num//2
    half_row_num=row_num//2
    img=haar_dwt(img,haar_wavelet_matrix_row,haar_wavelet_matrix_col)

    img[0:half_row_num,0:half_col_num]=Normalize(img[0:half_row_num,0:half_col_num])
    img[half_row_num:row_num,0:half_col_num]=Normalize(img[half_row_num:row_num,0:half_col_num])
    img[0:half_row_num,half_col_num:col_num]=Normalize(img[0:half_row_num,half_col_num:col_num])
    img[half_row_num:row_num,half_col_num:col_num]=Normalize(img[half_row_num:row_num,half_col_num:col_num])
    return img

```
由于图片的行列不在是2的次方倍数，且行列数不尽相同，根据上文提及的矩阵运算的过程，此时的运算过程:
$$
img_{pic}=
\begin{pmatrix}
x_{11}&x_{12}&\cdots&x_{1m}\\
x_{21}&\ddots&&\vdots\\
\vdots&&\ddots&\vdots\\
x_{n1}&\cdots&\cdots&x_{nm}\\
\end{pmatrix}
$$
$$
img_{row}=T_1 \cdot img_{pic}
$$
$$
img_{col}=T_2 \cdot (img_{row})^T
$$
$$
img_{pic}=(img_{col})^T
$$
即：
$$
img_{pic}= (T_2 \cdot (T_1 \cdot img_{pic})^T)^T=T_1\cdot img_{pic} \cdot T_2^T
$$
针对`Create_haar_matrix`和`haar_dwt`的代码要进行调整:
```python
def Create_haar_matrix(row_length,col_length):
    half_row_length=row_length//2
    half_col_length=col_length//2
    haar_wavelet_matrix_row=np.zeros((row_length,row_length),dtype=float)
    haar_wavelet_matrix_col=np.zeros((col_length,col_length),dtype=float)
    for i in range(half_row_length):
        haar_wavelet_matrix_row[i,i*2:i*2+1]=1/math.sqrt(2)
        haar_wavelet_matrix_row[half_row_length+i,i*2]=1/math.sqrt(2)
        haar_wavelet_matrix_row[half_row_length+i,i*2+1]=-1/math.sqrt(2)
    for i in range(half_col_length):
        haar_wavelet_matrix_col[i,i*2:i*2+1]=1/math.sqrt(2)
        haar_wavelet_matrix_col[half_col_length+i,i*2]=1/math.sqrt(2)
        haar_wavelet_matrix_col[half_col_length+i,i*2+1]=-1/math.sqrt(2)
    return haar_wavelet_matrix_col,haar_wavelet_matrix_row
def haar_dwt(img,haar_wavelet_matrix_row,haar_wavelet_matrix_col):
    Low_High_frequency=np.dot(np.dot(haar_wavelet_matrix_col,img),haar_wavelet_matrix_row.T)
    return Low_High_frequency
```
## 最终代码结果
```python
import numpy as np
import cv2
import math
# numpy数组归一化
def Normalize(img):
    _range = np.max(img) - np.min(img)
    return (img - np.min(img)) / _range
#根据输入数组长度创建小波变换矩阵
def Create_haar_matrix(row_length,col_length):
    half_row_length=row_length//2
    half_col_length=col_length//2
    haar_wavelet_matrix_row=np.zeros((row_length,row_length),dtype=float)
    haar_wavelet_matrix_col=np.zeros((col_length,col_length),dtype=float)
    for i in range(half_row_length):
        haar_wavelet_matrix_row[i,i*2:i*2+1]=1/math.sqrt(2)
        haar_wavelet_matrix_row[half_row_length+i,i*2]=1/math.sqrt(2)
        haar_wavelet_matrix_row[half_row_length+i,i*2+1]=-1/math.sqrt(2)
    for i in range(half_col_length):
        haar_wavelet_matrix_col[i,i*2:i*2+1]=1/math.sqrt(2)
        haar_wavelet_matrix_col[half_col_length+i,i*2]=1/math.sqrt(2)
        haar_wavelet_matrix_col[half_col_length+i,i*2+1]=-1/math.sqrt(2)
    return haar_wavelet_matrix_col,haar_wavelet_matrix_row

#inputdata: row or column of image
#outputdata: array handle with haar_wavelet
def haar_dwt(img,haar_wavelet_matrix_row,haar_wavelet_matrix_col):
    Low_High_frequency=np.dot(np.dot(haar_wavelet_matrix_col,img),haar_wavelet_matrix_row.T)
    return Low_High_frequency

#inputdata: array of image
#outputdata: array of image handel with haar_wavelet
def haar_dwt2D(img):
    col_num=img.shape[1]
    row_num=img.shape[0]
    if col_num%2!=0 and row_num%2!=0:
        img=cv2.copyMakeBorder(img,0,1,0,1,cv2.BORDER_REPLICATE)
    elif col_num%2==0 and row_num%2!=0:
        img=cv2.copyMakeBorder(img,0,1,0,0,cv2.BORDER_REPLICATE)
    elif col_num%2!=0 and row_num%2==0:
        img=cv2.copyMakeBorder(img,0,0,0,1,cv2.BORDER_REPLICATE)
    col_num=img.shape[1]
    row_num=img.shape[0]
    #创建小波变换矩阵
    haar_wavelet_matrix_row,haar_wavelet_matrix_col=Create_haar_matrix(row_num,col_num)
    half_col_num=col_num//2
    half_row_num=row_num//2
    img=haar_dwt(img,haar_wavelet_matrix_row,haar_wavelet_matrix_col)

    img[0:half_row_num,0:half_col_num]=Normalize(img[0:half_row_num,0:half_col_num])
    img[half_row_num:row_num,0:half_col_num]=Normalize(img[half_row_num:row_num,0:half_col_num])
    img[0:half_row_num,half_col_num:col_num]=Normalize(img[0:half_row_num,half_col_num:col_num])
    img[half_row_num:row_num,half_col_num:col_num]=Normalize(img[half_row_num:row_num,half_col_num:col_num])
    return img
if __name__ == '__main__':
    img= cv2.imread("pic1.png",0).astype(np.float64)
    img=haar_dwt2D(img)
    cv2.imshow('asd',img)
    cv2.waitKey(0)
```
注意考虑到代码的适用性，本文的代码仅可以压缩一次。若需要连续压缩，需要将需要压缩的图片数据单独取出，重新输入，考虑到代码的适用性，以后再做更新。即以下的形式不正确：
```pyhon
if __name__ == '__main__':
    #读取网络图片（2选1）
    # file_pic=requests.get('http://39.105.26.229:4567/pic.png')
    # img= cv2.imdecode(np.fromstring(file_pic.content, np.uint8), 0).astype(np.float64)
    #读取本地图片（2选1）
    img= cv2.imread("pic.png",0).astype(np.float64)
    img=haar_dwt2D(img)
    cv2.imshow('asd',img)
    cv2.waitKey(0)
    #!!!!!!!本文代码在此处不能照抄之前的运行代码，本文已经删除以下代码。
    for i in range(0,img.shape[0]//2+1,img.shape[0]//2):
        for j in range(0,img.shape[1]//2+1,img.shape[1]//2):
            img[i:i+img.shape[0]//2,j:j+img.shape[1]//2]=haar_dwt2D(img[i:i+img.shape[0]//2,j:j+img.shape[1]//2])
    cv2.imshow('asd',img)
    cv2.waitKey(0)  
```
## 最后的思考
>  本文采用haar wavelet对图像进行压缩，运算过程中将图像以行列作为区别，将二维图像转为一维矩阵运算，本质上应该还是属于一维的运算方法，但此方法感觉不应止于此。根据haar wavelet的计算原理，可否使用二维计算方法，即使用卷积计算。
>
> 计算思路如下:
> 
> 压缩后的每一个图像像素点仅与压缩前的2x2矩阵中的信息相关，所以针对这一特性，我们只需要给出4个不同的卷积核，分别对应于$LL,LH,HL,HH,$这四个图像。本文针对Haar wavelet,给出以下四个卷积矩阵：
> $$LL= {1\over 2}
\begin{pmatrix}
1&1\\
1&1\\
\end{pmatrix}
>$$
>$$HH= {1\over 2}
\begin{pmatrix}
1&-1\\
-1&1\\
\end{pmatrix}
>$$
> $$HL= {1\over 2}
\begin{pmatrix}
1&-1\\
1&-1\\
\end{pmatrix}
>$$
> $$HL= {1\over 2}
\begin{pmatrix}
1&1\\
-1&-1\\
\end{pmatrix}
>$$
>其中HL和LH的顺序可能不对，尚未验证。通过给出的四个卷积核，通过对图像进行卷积运算得出四个矩阵，即使haar wavelet的压缩结果。