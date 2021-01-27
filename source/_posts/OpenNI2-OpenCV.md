---
title: OpenNI2_OpenCV
date: 2020-12-06 09:24:37
mathjax: true
markup: mmark
categories:
  - 环境配置
tags:
  - Orbbec_OpenNI
  - OpenCV
  - Fanrencli
author: Fanrencli
---

## 奥比中光OpenNISDK安装（Linux）
奥比中光针对ZaroP1开发板和深度摄像机提供了相关的[OpenNI2的SDK](https://abzg-oss.oss-cn-shenzhen.aliyuncs.com/files/OpenNI-Linux-Arm64-2.3.0.65.rar),针对Windows/Linux/Android不同平台提供了相关的[安装文档](https://developer.orbbec.com.cn/technical_library.html?id=30).根据官方文档将OpenNI2配置完成，注意官方文档中说Linux平台内置驱动不需要额外安装，但是需要仔细阅读`README`文件：
```
OpenNI
------

Website: http://structure.io/openni

Building Prerequisites(前置依赖)
======================

Linux
-----
- GCC 4.x
    From: http://gcc.gnu.org/releases.html
    Or via apt:
    sudo apt-get install g++
- Python 2.6+/3.x
    From: http://www.python.org/download/
    Or via apt:
    sudo apt-get install python
- LibUSB 1.0.x
    From: http://sourceforge.net/projects/libusb/files/libusb-1.0/
    Or via apt:
    sudo apt-get install libusb-1.0-0-dev
- LibUDEV
    sudo apt-get install libudev-dev
- JDK 6.0
    From: http://www.oracle.com/technetwork/java/javase/downloads/jdk-6u32-downloads-1594644.html
    Or via apt:
    sudo apt-get install openjdk-6-jdk
- FreeGLUT3
    From: http://freeglut.sourceforge.net/index.php#download
    Or via apt:
    sudo apt-get install freeglut3-dev
- Doxygen
    From: http://www.stack.nl/~dimitri/doxygen/download.html#latestsrc
    Or via apt:
    sudo apt-get install doxygen
- GraphViz
    From: http://www.graphviz.org/Download_linux_ubuntu.php
    Or via apt:
    sudo apt-get install graphviz
```
将以上的依赖安装完成之后，运行其中示例代码还需要安装对应的USB驱动，驱动安装方法如下(此方法将同目录下的USB驱动文件拷贝到系统目录中，并自动生成环境文件指明OpenNI的路径)：

进入安装包OpenNI目录中
```sh
 orbbec@localhost:OpenNI$ ./install.sh
```
最后，在安装文件中找到`NiViewer`运行文件，若文件不能运行，查看`chmod`权限。
```sh
sudo ./NiViewer
```
![结果图片](https://api.orbbec.com.cn/uploads/kindeditor/20200718113033.jpg)
经过以上的配置过程，OpenNI的环境就配置完成了。

## OpenCV在Arm平台上编译
在运用开发板设备获取数据的时候，通常运用`OpenNI2`获取数据流，通过`OpenCV`对数据流进行转换，生成RGB图片和深度图片。

### 下载源文件
源文件[官方地址](https://opencv.org/releases/)，本文用的是`OpenCV3.4.3`版本。OpenCV编译的方法自行百度，编译完成后对orbbec中OpenNI2SDK和opencv进行配置。

## OpenCV+OpenNI2配置
根据官方给出的示例代码，进行编写。由于官方的代码是在`Makefile`文件中进行编写，所以opencv也需要在其中编写。根据`Makefile`文件编写规则进行编写。
```Makefile
#Includes
CFLAGS = ${shell pkg-config opencv --cflags}

#lib
LDFLAGS = ${shell pkg-config opencv --libs}

```
其中`pkg-config`需要在系统中配置`opencv.pc`文件，文件内容，在安装的文件中可以找到，若没有则可能默认没有生成（opencv4之后的版本默认不生成），需要在编译的时候设置。

`opencv`环境配置没有问题后，在orbbec提供的OpenNI2的文件中示例代码中进行整合开发。文件位置在：
```sh
/OpenNI/Samples/
```
选择其中`SimpleViewer`文件夹中的示例代码进行编写。在`CommonCppMakefile`文件夹中找到`CFLAGS`和`LDFLAGS`变量位置，在其后加上：
```Makefile
#Includes
CFLAGS += ${shell pkg-config opencv --cflags}

#lib
LDFLAGS += ${shell pkg-config opencv --libs}

```
此外还需要在`Makefile`文件中加入OpenNI头文件的路径和Redist的路径，这就是在配置OpenNI时运行`install.sh`文件生成的环境文件中的内容，也在同级目录下。

其后编译即可通过。接下来就可以使用opencv对数据进行获取保存了。