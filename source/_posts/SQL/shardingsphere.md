---
title: ShardingSphere
date: 2025-09-20 12:58:00
cover: true
top: true
categories:
  - SQL
tags:
  - Mysql
author: Fanrencli
---

## ShardingSphere

### 1. 简介

Apache ShardingSphere 是一套开源的分布式数据库中间件解决方案组成的生态圈，它由 JDBC、Proxy 和 Sidecar（规划中）这 3 款相互独立，却又能够混合部署使用的多款产品组成。它们均提供标准化的数据分片、分布式事务和分布式治理功能，可适用于如 Java 同构、异构语言、云原生等各种多样化的应用场景。

数据库分片架构主要包括：垂直分库、垂直分表、水平分库、水平分表。

- 垂直分库：根据业务耦合性，将关联度低的不同表存储在不同的数据库。通过将数据库拆分，提升IO、降低单机并发量及单机硬件成本。例如：用户表和订单表在业务上基本没有耦合性，可以将用户表和订单表进行分库。
- 垂直分表：基于分库的思想，将单库的表拆分成多张表，每张表只存储一部分数据。例如：将Age和Name字段放在一张表，将Address字段放在另一张表。
- 水平分库：将单库的数据，按照一定规则拆分到多个数据库中。
- 水平分表：将单表的记录，按照一定规则拆分到多个表中。

根据上述的数据库架构，衍生而来的问题就是在业务中如何对数据库进行读写操作，如何保证数据的一致性，如何进行分布式事务处理等。常用的实现方式主要分为：程序代码封装和中间件封装。

- 程序代码封装：通过在代码中封装分库分表、读写分离、分布式事务等操作，实现数据库的读写操作。优点是简单，缺点是代码耦合度高，不利于维护。
- 中间件封装：通过引入中间件，实现数据库的读写操作。优点是代码耦合度低，易于维护，缺点是引入中间件，增加了系统的复杂度。

本文介绍的shardingSphere，就是一款包括程序级别封装和中间件封装的数据库解决方案。

### 2. ShardingSphere-JDBC

ShardingSphere-JDBC 是 Apache ShardingSphere 的第一个产品，也是 Apache ShardingSphere 的前身，它定位为轻量级 Java 框架，在 Java 的 JDBC 层提供的额外服务。 它使用客户端直连数据库，以 jar 包形式提供服务，无需额外部署和依赖，可理解为增强版的 JDBC 驱动，完全兼容 JDBC 和各种 ORM 框架。

![ShardingSphere-JDBC架构](http://fanrencli.cn/fanrencli.cn/shardingsphere1.png)

#### 2.1 读写分离实践




### 3. ShardingSphere-Proxy

ShardingSphere-Proxy 是 Apache ShardingSphere 的第二个产品，它定位为透明化的数据库代理端，提供封装了数据库二进制协议的服务端版本，用于完成对异构语言的支持。 目前提供 MySQL 和 PostgreSQL 版本，它可以使用任何兼容 MySQL/PostgreSQL 协议的访问客户端（如：MySQL Command Client, MySQL Workbench, Navicat 等）操作数据，对 DBA 更加友好。


![ShardingSphere-Proxy架构](http://fanrencli.cn/fanrencli.cn/shardingsphere2.png)

#### 3.1 数据分片