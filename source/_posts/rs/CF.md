---
title: Collaborative Filtering
date: 2025-09-20 12:58:00
mathjax: true
markup: mmark
cover: true
top: true
categories:
  - Recommender System
tags:
  - RS
  - ReCall
author: Fanrencli
---

## 协同过滤(Collaborative Filtering)

### 基本原理

协同过滤（Collaborative Filtering）推荐算法是最经典、最常用的推荐算法。基本思想是：

- 根据用户之前的喜好以及其他兴趣相近的用户的选择来给用户推荐物品。
    - 基于对用户历史行为数据的挖掘发现用户的喜好偏向， 并预测用户可能喜好的产品进行推荐。
    - 一般是仅仅基于用户的行为数据（评价、购买、下载等）, 而不依赖于项的任何附加信息（物品自身特征）或者用户的任何附加信息（年龄， 性别等）。
- 目前应用比较广泛的协同过滤算法是基于邻域的方法，主要有：
    - 基于用户的协同过滤算法（UserCF）：给用户推荐和他兴趣相似的其他用户喜欢的产品。
    - 基于物品的协同过滤算法（ItemCF）：给用户推荐和他之前喜欢的物品相似的物品。

不管是 UserCF 还是 ItemCF 算法， 重点是计算用户之间（或物品之间）的相似度。

### 相似性度量

1. 杰卡德（Jaccard）相似系数

Jaccard 系数是衡量两个集合的相似度一种指标，计算公式如下：

$$sim_{uv}=\frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$$

- 其中$N(u)$，$N(v)$分别表示用户$u$和用户$v$交互物品的集合。
- 对于用户$u$和$v$，该公式反映了两个交互物品交集的数量占这两个用户交互物品并集的数量的比例。

由于杰卡德相似系数一般无法反映具体用户的评分喜好信息，所以常用来评估用户是否会对某物品进行打分， 而不是预估用户会对某物品打多少分。


2. 余弦相似度

余弦相似度衡量了两个向量的夹角，夹角越小越相似。余弦相似度的计算如下，其与杰卡德（Jaccard）相似系数只是在分母上存在差异：

$$sim_{uv}=\frac{|N(u)\cap N(v)|}{\sqrt{|N(u)|\cdot|N(v)|}}$$

从向量的角度进行描述，令矩阵 $A$ 为用户-物品交互矩阵，矩阵的行表示用户，列表示物品。

- 设用户和物品数量分别为 $m$，$n$，交互矩阵 $A$ 就是一个 $m$ 行 $n$ 列的矩阵。
- 矩阵中的元素均为 $0/1$。若用户 $i$ 对物品 $j$ 存在交互，那么 $A_{i,j}=1$，否则为 $0$。
- 那么，用户之间的相似度可以表示为：

$$sim_{uv}=cos(u,v)=\frac{u \cdot v}{|u|\cdot|v|}$$

- 向量 $u$，$v$ 在形式都是 one-hot 类型，$u\cdot v$ 表示向量点积。

上述用户-物品交互矩阵在现实中是十分稀疏的，为了节省内存，交互矩阵会采用字典进行存储。在 sklearn 中，余弦相似度的实现：
``` python
from sklearn.metrics.pairwise import cosine_similarity

i = [1, 0, 0, 0]
j = [1, 0, 1, 0]
cosine_similarity([i, j])
```

3. 皮尔逊相关系数

在用户之间的余弦相似度计算时，将用户向量的内积展开为各元素乘积和：

$$sim_{uv}=\frac{\sum_{i}r_{ui} * r_{vi}}{\sqrt{\sum_{i}r_{ui}^{2}}\sqrt{\sum_{i}r_{vi}^{2}}}$$

- 其中，$r_{ui}$，$r_{vi}$ 分别表示用户 $u$ 和用户 $v$ 对物品 $i$ 是否有交互（或具体评分值）。

皮尔逊相关系数与余弦相似度的计算公式非常类似，如下：

$$
\operatorname{sim}(u, v)=\frac{\sum_{i \in I}\left(r_{u i}-\bar{r}_{u}\right)\left(r_{v i}-\bar{r}_{v}\right)}{\sqrt{\sum_{i \in I}\left(r_{u i}-\bar{r}_{u}\right)^{2}} \sqrt{\sum_{i \in I}\left(r_{v i}-\bar{r}_{v}\right)^{2}}}
$$

- 其中，$r_{ui}$，$r_{vi}$ 分别表示用户 $u$ 和用户 $v$ 对物品 $i$ 是否有交互（或具体评分值）；
- $\bar{r}_u$，$\bar{r}_v$ 分别表示用户 $u$ 和用户 $v$ 交互的所有物品交互数量或者评分的平均值；

相较于余弦相似度，皮尔逊相关系数通过使用用户的平均分对各独立评分进行修正，减小了用户评分偏置的影响。在 scipy 中，皮尔逊相关系数的实现：
```python
from scipy.stats import pearsonr

i = [1, 0, 0, 0]
j = [1, 0.5, 0.5, 0]
pearsonr(i, j)
```

### 代码实战

``` python 

import numpy as np 
import pandas as pd

def loadData(): 
    users = {'Alice': {'A': 5, 'B': 3, 'C': 4, 'D': 4}, 'user1': {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3}, 'user2': {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5}, 'user3': {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4}, 'user4': {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1} } 
    return users

user_data = loadData()
similarity_matrix = pd.DataFrame(
    np.identity(len(user_data)),
    index=user_data.keys(),
    columns=user_data.keys(),
)

# 遍历每条用户-物品评分数据
for u1, items1 in user_data.items():
    for u2, items2 in user_data.items():
        if u1 == u2:
            continue
        vec1, vec2 = [], []
        for item, rating1 in items1.items():
            rating2 = items2.get(item, -1)
            if rating2 == -1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)
        # 计算不同用户之间的皮尔逊相关系数
        similarity_matrix[u1][u2] = np.corrcoef(vec1, vec2)[0][1]

print(similarity_matrix)


target_user = ' Alice '
num = 2
# 由于最相似的用户为自己，去除本身
sim_users = similarity_matrix[target_user].sort_values(ascending=False)[1:num+1].index.tolist()
print(f'与用户{target_user}最相似的{num}个用户为：{sim_users}')


weighted_scores = 0.
corr_values_sum = 0.

target_item = 'E'
# 基于皮尔逊相关系数预测用户评分
for user in sim_users:
    corr_value = similarity_matrix[target_user][user]
    user_mean_rating = np.mean(list(user_data[user].values()))

    weighted_scores += corr_value * (user_data[user][target_item] - user_mean_rating)
    corr_values_sum += corr_value

target_user_mean_rating = np.mean(list(user_data[target_user].values()))
target_item_pred = target_user_mean_rating + weighted_scores / corr_values_sum
print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred}')
```


### 问题

User-based算法存在两个重大问题：

- 数据稀疏性
    - 一个大型的电子商务推荐系统一般有非常多的物品，用户可能买的其中不到1%的物品，不同用户之间买的物品重叠性较低，导致算法无法找到一个用户的邻居，即偏好相似的用户。
    - 这导致UserCF不适用于那些正反馈获取较困难的应用场景(如酒店预订， 大件物品购买等低频应用)。
- 算法扩展性
    - 基于用户的协同过滤需要维护用户相似度矩阵以便快速的找出TopN相似用户，该矩阵的存储开销非常大，存储空间随着用户数量的增加而增加。故不适合用户数据量大的情况使用。

由于UserCF技术上的两点缺陷，导致很多电商平台并没有采用这种算法，而是采用了ItemCF算法实现最初的推荐系统。

### 算法评估

#### 召回率

对用户 $u$ 推荐 $N$ 个物品记为 $R(u)$，令用户 $u$ 在测试集上喜欢的物品集合为 $T(u)$，那么召回率定义为：

$$ \text{Recall}=\frac{\sum_{u}|R(u)\cap T(u)|}{\sum_{u}|T(u)|} $$

- 含义：在模型召回预测的物品中，预测准确的物品占用户实际喜欢的物品的比例。

#### 精确率

精确率定义为：

$$ \text{Precision}=\frac{\sum_{u}|R(u)\cap T(u)|}{\sum_{u}|R(u)|} $$

- 含义：推荐的物品中，对用户准确推荐的物品占总物品的比例。

- 如要确保召回率高，一般是推荐更多的物品，期望推荐的物品中会涵盖用户喜爱的物品。而实际中，推荐的物品中用户实际喜爱的物品占少数，推荐的精确率就会很低。故同时要确保高召回率和精确率往往是矛盾的，所以实际中需要在二者之间进行权衡。

#### 覆盖率

覆盖率反映了推荐算法发掘长尾的能力，覆盖率越高，说明推荐算法越能将长尾中的物品推荐给用户。

$$ \text{Coverage} = \frac{\left| \bigcup_{u \in U} R(u) \right|}{|I|} $$

- **含义**：推荐系统能够推荐出来的物品占总物品集合的比例。
  - 其中 $|I|$ 表示所有物品的个数；
  - 系统的用户集合为 $U$；
  - 推荐系统给每个用户推荐一个长度为 $N$ 的物品列表 $R(u)$。

- 覆盖率表示最终的推荐列表中包含多大比例的物品。如果所有物品都被给推荐给至少一个用户，那么覆盖率是 $100\%$。