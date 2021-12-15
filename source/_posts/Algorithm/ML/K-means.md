---
title: K-means
date: 2021-10-16 23:20:42
categories:
  - Algorithm
tags:
  - python
  - Cluster
author: Fanrencli
---
### 聚类算法 —— K-means

####算法原理

- 在给定的数据集上随机选定`K`个种子点
- 对数据集中每个样本与选定的种子点之间的距离进行计算
- 将距离最近的点归纳到对应的种子点所代表的聚类中
- 针对`k`个聚类重新计算每个类别的质点作为新的种子点
- 重复上述的距离计算，并更新种子点，直至不再改变

#### 算法优劣

- `k-means`算法需要具备一定的先验知识，选择聚类的数量决定最终的结果
- 聚类的结果还受到随机点选择的影响
- 无监督训练，对异常值较为敏感
- 对样本分布有一定的要求，样本不能分布太过离散，对非凸形状的分类效果不好

#### 代码实现
```python
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist=[]
    for data in dataSet:
        diff = np.tile(data, (k, 1)) - centroids  #相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2     #平方
        squaredDist = np.sum(squaredDiff, axis=1)   #和  (axis=1表示行)
        distance = squaredDist ** 0.5  #开根号
        clalist.append(distance) 
    clalist = np.array(clalist)  #返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist

# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)    #axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean() #DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values
 
    # 计算变化量
    changed = newCentroids - centroids
 
    return changed, newCentroids

# 使用k-means分类
def kmeans(dataSet, k):
    # 随机取质心
    centroids = random.sample(dataSet, k)
    
    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)
 
    centroids = sorted(newCentroids.tolist())   #tolist()将矩阵转换成列表 sorted()排序
 
    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k) #调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)  
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):   #enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])
        
    return centroids, cluster
 
# 创建数据集
def createDataSet():
    return [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]

if __name__=='__main__': 
    dataset = createDataSet()
    centroids, cluster = kmeans(dataset, 2)
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)
    for i in range(len(dataset)):
      plt.scatter(dataset[i][0],dataset[i][1], marker = 'o',color = 'green', s = 40 ,label = '原始点')
                                                    #  记号形状       颜色      点的大小      设置标签
      for j in range(len(centroids)):
        plt.scatter(centroids[j][0],centroids[j][1],marker='x',color='red',s=50,label='质心')
        plt.show
```