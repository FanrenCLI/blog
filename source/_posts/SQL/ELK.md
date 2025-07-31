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


- 插入文档：PUT、POST /index_name/type_name/_doc/{id}，请求体为 JSON 格式的数据

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

- 更新文档：PUT、POST /index_name/type_name/_update/{id}，请求体为 JSON 格式的数据



## Logstash

## Logs


## Kibana


## Beats
