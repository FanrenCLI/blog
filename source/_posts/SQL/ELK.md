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

- 查询所有索引数据： GET /_cat/indices?v

```http
PUT http://localhost:9200/_cat/indices?v

Response:
health status index     uuid                   pri rep docs.count docs.deleted store.size pri.store.size
green  open   shopping EUd-Fh97S8WkTG5VIkV6TA   1   0          0            0       225b           225b
```


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
| `_cat`            | **监控命令入口** (含60+子端点) | `GET /_cat/indices`    |
| `_cluster`        | 集群操作      | `GET /_cluster/health`|
| `_nodes`          | 节点管理      | `GET /_nodes/stats`   |
| `_tasks`          | 管理异步任务  | `GET /_tasks?detailed=true`           |

5. 安全类（这个在原回答中只有两列，没有示例列）：
| **关键词**        | **作用**       |
|-------------------|---------------------|
| `_security`       | 用户/角色权限管理    |
| `_api_key`        | API密钥管理   |
| `_oidc`           | OpenID Connect认证  |

### 代码实战

- 依赖引入

```xml
<!-- 高级客户端，es8+之后推荐 -->
<!-- https://mvnrepository.com/artifact/co.elastic.clients/elasticsearch-java -->
<dependency>
    <groupId>co.elastic.clients</groupId>
    <artifactId>elasticsearch-java</artifactId>
    <version>9.1.0</version>
</dependency>
<!-- 高级客户端，不过只适用于es7+ -->
<!-- https://mvnrepository.com/artifact/org.elasticsearch.client/elasticsearch-rest-high-level-client -->
<dependency>
    <groupId>org.elasticsearch.client</groupId>
    <artifactId>elasticsearch-rest-high-level-client</artifactId>
    <version>7.17.29</version>
</dependency>
<!-- 低级客户端，基础的线程管理和http访问 -->
<!-- https://mvnrepository.com/artifact/org.elasticsearch.client/elasticsearch-rest-client -->
<dependency>
    <groupId>org.elasticsearch.client</groupId>
    <artifactId>elasticsearch-rest-client</artifactId>
    <version>9.1.0</version>
</dependency>
<!-- 如果使用springboot项目，使用此依赖可以通过spring进行管理，通过注解简化开发 -->
<dependency>
    <groupId>org.springframework.data</groupId>
    <artifactId>spring-data-elasticsearch</artifactId>
    <version>4.4.18</version>
</dependency>
```

- 代码示例

```java
package com.example.es;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.Refresh;
import co.elastic.clients.elasticsearch._types.SortOptions;
import co.elastic.clients.elasticsearch._types.SortOrder;
import co.elastic.clients.elasticsearch.core.*;
import co.elastic.clients.elasticsearch.core.search.Hit;
import co.elastic.clients.elasticsearch.indices.CreateIndexResponse;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@SpringBootTest
class EsApplicationTests {
    static class Product {
        public String id;
        public String name;
        public double price;
        public String category;

        Product() {

        }

        public Product(String id, String name, double price, String category) {
            this.id = id;
            this.name = name;
            this.price = price;
            this.category = category;
        }

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public double getPrice() {
            return price;
        }

        public void setPrice(double price) {
            this.price = price;
        }

        public String getCategory() {
            return category;
        }

        public void setCategory(String category) {
            this.category = category;
        }
    }

    public static ElasticsearchClient client;

    @Test
    public void testEs() throws Exception {
        RestClient restClient = RestClient.builder(
                new HttpHost("localhost", 9200, "http")
        ).build();
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
        RestClientTransport restClientTransport = new RestClientTransport(restClient, new JacksonJsonpMapper(objectMapper));
        client = new ElasticsearchClient(restClientTransport);
        createProductIndex();

        Product product = new Product("p1", "Laptop", 1299.99, "Electronics");
        indexProduct(product);     // 创建文
        getProduct("p1");          // 获取文档
        updateProductPrice("p1", 1199.99); // 更新文档
        searchProducts("Laptop");  // 搜索文档
        searchAllProducts();
        deleteProduct("p1");       // 删除文档档
        restClient.close();
    }

    // ========= 索引操作 =========
    private static void createProductIndex() throws IOException {
        // 检查索引是否存在
        boolean exists = client.indices()
                .exists(e -> e.index("products"))
                .value();

        if (!exists) {
            // 创建索引并配置映射
            CreateIndexResponse response = client.indices().create(c -> c
                    .index("products")
                    .mappings(m -> m
                            .properties("id", p -> p.keyword(k -> k))
                            .properties("name", p -> p.text(k -> k))
                            .properties("price", p -> p.text(k -> k))
                            .properties("category", p -> p.text(k -> k))
                    )
            );
            System.out.println("索引创建结果: " + response.acknowledged());
        }
    }

    // ========= 文档操作 =========
    // 1. 索引文档 (UPSERT)
    private static void indexProduct(Product product) throws IOException {
        IndexResponse response = client.index(i -> i
                .index("products")
                .id(product.getId())
                .document(product)
        );
        System.out.println("索引文档结果: " + response.result());
    }

    // 2. 获取文档
    private static void getProduct(String id) throws IOException {
        GetResponse<Product> response = client.get(g -> g
                        .index("products")
                        .id(id),
                Product.class
        );

        if (response.found()) {
            Product product = response.source();
            System.out.println("获取到的产品: " + product.getName() + ", 价格: $" + product.getPrice());
        } else {
            System.out.println("文档不存在");
        }
    }

    // 3. 更新文档 (部分更新)
    private static void updateProductPrice(String id, double newPrice) throws IOException {
        Map<String, Object> updatefield = new HashMap<>();
        updatefield.put("price", newPrice);
        UpdateResponse<Product> response = client.update(u -> u
                        .index("products")
                        .id(id)
                        .doc(updatefield)
                        .refresh(Refresh.True),
                Product.class
        );
        System.out.println("更新结果: " + response.result());
    }

    // 4. 搜索文档
    private static void searchProducts(String keyword) throws IOException {
        SearchResponse<Product> response = client.search(s -> s
                        .index("products")
                        .query(q -> q
                                .match(m -> m
                                        .field("name")
                                        .query(keyword)
                                )
                        )
                        .from(0)
                        .size(10)
                        .sort(SortOptions.of(so -> so
                                        .field(f -> f
                                                .field("price")
                                                .order(SortOrder.Desc)
                                        )
                                )
                        )
                        .source(sc -> sc
                                .filter( f->f
                                        .excludes("price")
                                        .includes("name"))
                        ),
                Product.class
        );
        System.out.println("====== 搜索结果 =====");
        System.out.println("总命中数: " + response.hits().total().value());
        for (Hit<Product> hit : response.hits().hits()) {
            Product p = hit.source();
            System.out.printf("[%s] %s - $%.2f\n",
                    p.getId(), p.getName(), p.getPrice());
        }
    }

    // 5. 删除文档
    private static void deleteProduct(String id) throws IOException {
        DeleteResponse response = client.delete(d -> d
                .index("products")
                .id(id)
        );
        System.out.println("删除结果: " + response.result());
    }

    // 6. 搜索文档
    private static void searchAllProducts() throws IOException {
        SearchResponse<Product> response = client.search(s -> s
                        .index("products")
                        .query(q -> q
                                .matchAll(m -> m)
                        ),
                Product.class
        );
        System.out.println("====== 全量查询结果 =====");
        System.out.println("总命中数: " + response.hits().total().value());
        for (Hit<Product> hit : response.hits().hits()) {
            Product p = hit.source();
            System.out.printf("[%s] %s - $%.2f\n",
                    p.getId(), p.getName(), p.getPrice());
        }
    }
}

```


## Logstash

## Logs


## Kibana


## Beats
