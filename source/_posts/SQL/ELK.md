---
title: ELK
date: 2025-07-19 18:00:00
cover: true
top: true
categories:
  - DB
  - ElasticSearch
  - Logstash
  - Kibana
  - Beats
tags:
  - Java
author: Fanrencli
---

# ELK

## 简介

ELK 是 Elasticsearch、Logstash、Kibana 的简称，它们都是开源软件，通常一起使用来收集、存储、搜索和分析数据。

- **Elasticsearch**：一个分布式搜索和分析引擎，用于存储和搜索大量数据。它提供了全文搜索、结构化搜索、分析等功能。
- **Logstash**：一个数据处理管道，用于收集、处理和转发日志和事件数据。它可以从各种来源收集数据，并进行格式转换、过滤、解析等操作。然后将数据发送到 Elasticsearch。
- **Kibana**：一个数据可视化工具，用于创建仪表板和可视化数据。它可以与 Elasticsearch 集成，提供丰富的数据可视化功能，帮助用户更好地理解数据。
- **Beats**：轻量级的数据收集器，用于将数据发送到 Logstash 或 Elasticsearch。Beats 包括 Filebeat（用于收集日志文件）、Metricbeat（用于收集系统和服务指标）等。

## ElasticSearch

Elasticsearch 是一个分布式搜索和分析引擎，用于存储和搜索大量数据。它提供了全文搜索、结构化搜索、分析等功能。Elasticsearch 使用 JSON 格式存储数据，并使用倒排索引来加速搜索。所谓倒排索引，就是将文档中的每个词映射到包含该词的文档列表，通过查询关键词，可以快速找到包含该关键词的文档。

### 数据格式

- index：索引，类似于数据库中的数据库。
- type：类型，类似于数据库中的表。
- document：文档，类似于数据库中的行。
- field：字段，类似于数据库中的列。

### 数据库操作

- 创建索引：PUT /index_name

```http
PUT http://localhost:9200/shopping

Response:
{
  "acknowledged": true,
  "shards_acknowledged": true,
  "index": "sleep"
}
```

- 删除索引：DELETE /index_name

```http
DELETE http://localhost:9200/shopping

Response:
{
  "acknowledged": true
}
```

- 查询索引：GET /index_name

```http
GET http://localhost:9200/shopping

Response:
{
  "shopping": {
    "aliases": {},
    "mappings": {},
    "settings": {
      "index": {
        "routing": {
          "allocation": {
            "include": {
              "_tier_preference": "data_content"
            }
          }
        },
        "number_of_shards": "1",
        "provided_name": "shopping",
        "creation_date": "1753983017737",
        "number_of_replicas": "1",
        "uuid": "EUd-Fh97S8WkTG5VIkV6TA",
        "version": {
          "created": "9009000"
        }
      }
    }
  }
}
```


- 插入更新文档：PUT、POST /index_name/_doc/{id}，请求体为 JSON 格式的数据,uri中必须要有$_doc$

```http
POST http://localhost:9200/shopping/_doc/1
Content-Type: application/json

{
    "title": "Macbook Pro",
    "price": 2000,
    "description": "Macbook Pro 16 inch"
}

Response:
{
  "_index": "shopping",
  "_id": "1",
  "_version": 1,
  "result": "created",
  "_shards": {
  },
  "_seq_no": 1,
  "_primary_term": 1
}
```

- 部分更新文档：POST /index_name/_update/{id}，请求体为 JSON 格式的数据,一定要有$doc$

```http

POST http://localhost:9200/shopping/_update/1
Content-Type: application/json

{
    "doc": {
        "title": "Macbook Pro 1",
        "price": 2000,
        "description": "Macbook Pro 16 inch"
    }
}
```

- PUT为幂等操作，POST为非幂等操作，PUT会覆盖原有数据，使用时必须指定id，POST会自动生成id，不需要指定id，如果指定id那就是更新。PUT为全量更新替换,POST通过_update进行部分更新。

- 查询操作：GET /index_name/_doc/{id}

```http
GET http://localhost:9200/shopping/_search
Content-type: application/json

{
    "query": {
        "match": {
            "title": "MacBook"
        }
    },
    "from": 0,
    "size": 10,
    "sort": [
        {
            "price": {
                "order": "desc"
            }
        }
    ],
    "_source": ["title", "price"],
    "highlight": {
        "fields": {
            "title": {}
        }
    }
}
###

GET http://localhost:9200/shopping/_search
Content-type: application/json

{
    "aggs":{ # 聚合查询
        "price_group":{ # 查询名称随意即可
            "terms":{ # 聚合类型-分组
                "field": "price" # 聚合字段
            }
        }
    },
    "size":0 # 不返回数据
}
```

### 核心API总结


| **操作** | **方法** | **端点** | **文档 ID 来源** | **结果 (ID 不存在)** | **结果 (ID 存在)** | **主要用途** | **更新类型** |
| :------------------------ | :----- | :-------------------------- | :--------------- | :-------------------- | :---------------------- | :--------------- | :--------- |
| 创建 (自动 ID)          | `POST` | `/<index>/_doc`             | ES 自动生成      | **创建** 新文档       | (不适用 - 总是新 ID)    | 新增日志事件、未知 ID 的数据导入             | -          |
| 创建或**全量替换**       | `PUT`  | `/<index>/_doc/<id>`        | **用户明确指定** | **创建** 新文档       | **替换** 整个现有文档   | 提供完整 ID 时的索引、强制覆盖旧文档版本     | **全量替换** |
| **强制创建** (仅当不存在) | `PUT`/`POST`  | `/<index>/_create/<id>`     | **用户明确指定** | **创建** 新文档       | **失败** (409 Conflict) | 确保不覆盖的初始化、“仅当不存在时”操作       | -          |
| **部分更新**             | `POST` | `/<index>/_update/<id>`     | **用户明确指定** | **失败** (404 Not Found) | **更新** 指定字段       | 修改文档的特定部分（如计数器、状态标记、描述） | **部分更新** |


1. 文档操作类：
| **关键词**        | **作用**             | **示例**      |
|-------------------|---------------------|---------------|
| `_doc`            | 基础文档操作端点（GET查询，PUT/POST创建文档）| `PUT /index/_doc/1`      |
| `_create`         | 强制创建（仅当ID不存在）    | `PUT /index/_create/1`     |
| `_update`         | 部分更新文档          | `POST /index/_update/1`   |
| `_bulk`           | 批量操作API              | `POST /_bulk`         |
| `_source`         | 获取文档原始内容     | `GET /index/_source/1`     |
| `_mget`           | 批量获取多文档       | `GET /index/_mget`    |

2. 索引管理类：
| **关键词**        | **作用**               | **示例**        |
|-------------------|---------------------|------------|
| `_mapping`        | 管理字段映射  | `PUT /index/_mapping` |
| `_settings`       | 管理索引配置  | `PUT /index/_settings`|
| `_alias`          | 操作索引别名  | `POST /_aliases`      |
| `_forcemerge`     | 强制合并段文件| `POST /index/_forcemerge`    |

3. 搜索查询类：
| **关键词**        | **作用**       | **示例**               |
|-------------------|---------------------|------------|
| `_search`         | 执行搜索请求  | `GET /index/_search`  |
| `_count`          | 统计匹配文档数| `GET /index/_count`   |
| `_explain`        | 解释文档相关性评分    | `GET /index/_explain/1`    |
| `_knn_search`     | 向量相似度搜索 (8.0+)   | `POST /index/_knn_search`   |

4. 集群监控类：
| **关键词**        | **作用**       | **示例**               |
|-------------------|---------------------|------------|
| `_cat`            | **监控命令入口** (含60+子端点) | `GET /_cat/health`    |
| `_cluster`        | 集群操作      | `GET /_cluster/health`|
| `_nodes`          | 节点管理      | `GET /_nodes/stats`   |
| `_tasks`          | 管理异步任务  | `GET /_tasks?detailed=true`           |

5. 安全类（这个在原回答中只有两列，没有示例列）：
| **关键词**        | **作用**       |
|-------------------|---------------------|
| `_security`       | 用户/角色权限管理              |
| `_api_key`        | API密钥管理   |
| `_oidc`           | OpenID Connect认证            |

### 代码实战

## Logstash

## Logs


## Kibana


## Beats
