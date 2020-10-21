---
title: Hello World
cover: true
top: true
img: http://static.blinkfox.com/20181105-io.jpg
categories:
  - 前端
  - 后端
tags:
  - JavaScript
  - JAVA
author: DuAo
---
## 猪体朝向
>#### 数据准备
>>1. 猪体单测点云数据
>>2. 猪体对应的rgb图像
>>3. 猪体对应的深度图像
>
>#### 处理流程
>>**针孔相机模型**
>>![camera](../image/camera.png)
>>1. 世界坐标系(**世界坐标系主要用于不同点云的配准**)
>>2. 相机坐标系(**点云可视化时的默认坐标系**)
>>3. 图像坐标系(**以图像中心为原点**)
>>4. 像素坐标系(**以左上角为原点**)
>
>
>>**本文处理的数据主要为从像素坐标系到相机坐标系**
>>
>>![coordinateTrans](../image/coordinateTrans.png)
>>其中相机内参主要用于像素坐标系转为相机坐标系，相机外参用于相机坐标系转世界坐标系
>>$f$<sub>$x$</sub>=$f/dx$;$f$<sub>$y$</sub>=$f/dy$;$u$<sub>0</sub>,$v$<sub>0</sub>为相机光心在图像上的像素坐标