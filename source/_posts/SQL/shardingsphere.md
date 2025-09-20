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

- 引入依赖

```gradle
plugins {
    id 'java'
    id 'org.springframework.boot' version '2.7.6'
    id 'io.spring.dependency-management' version '1.1.7'
}

group = 'com.example'
version = '0.0.1-SNAPSHOT'
description = 'shardingsphere'

dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    implementation("com.baomidou:mybatis-plus-boot-starter:3.5.5")
    implementation("org.apache.shardingsphere:shardingsphere-jdbc-core-spring-boot-starter:5.1.0")
    compileOnly 'org.projectlombok:lombok'
    developmentOnly 'org.springframework.boot:spring-boot-devtools'
    runtimeOnly 'mysql:mysql-connector-java'
    annotationProcessor 'org.projectlombok:lombok'
}
```

- 配置文件

```application.properties
# 应用名称
spring.application.name=sharging-jdbc-demo


# 配置真实数据源
spring.shardingsphere.datasource.names=master,slave1

# 配置第 1 个数据源
spring.shardingsphere.datasource.master.type = com.zaxxer.hikari.HikariDataSource
spring.shardingsphere.datasource.master.driver-class-name = com.mysql.cj.jdbc.Driver
spring.shardingsphere.datasource.master.jdbc-url = jdbc:mysql://106.14.135.70:13306/user_db
spring.shardingsphere.datasource.master.username=root
spring.shardingsphere.datasource.master.password=123456

# 配置第 2 个数据源
spring.shardingsphere.datasource.slave1.type=com.zaxxer.hikari.HikariDataSource
spring.shardingsphere.datasource.slave1.driver-class-name=com.mysql.cj.jdbc.Driver
spring.shardingsphere.datasource.slave1.jdbc-url=jdbc:mysql://106.14.135.70:23306/user_db
spring.shardingsphere.datasource.slave1.username=root
spring.shardingsphere.datasource.slave1.password=123456


# 读写分离类型，如: Static，Dynamic
spring.shardingsphere.rules.readwrite-splitting.data-sources.myds.type=Static
# 写数据源名称
spring.shardingsphere.rules.readwrite-splitting.data-sources.myds.props.write-data-source-name=master
# 读数据源名称，多个从数据源用逗号分隔
spring.shardingsphere.rules.readwrite-splitting.data-sources.myds.props.read-data-source-names=slave1

# 负载均衡算法名称
spring.shardingsphere.rules.readwrite-splitting.data-sources.myds.load-balancer-name=alg_round

# 负载均衡算法配置
# 负载均衡算法类型
spring.shardingsphere.rules.readwrite-splitting.load-balancers.alg_round.type=ROUND_ROBIN
spring.shardingsphere.rules.readwrite-splitting.load-balancers.alg_random.type=RANDOM
spring.shardingsphere.rules.readwrite-splitting.load-balancers.alg_weight.type=WEIGHT
spring.shardingsphere.rules.readwrite-splitting.load-balancers.alg_weight.props.slave1=1

# 打印SQl
spring.shardingsphere.props.sql-show=true

```

- 代码实现

```java

@RestController("/test")
public class testcontroller {

    @Autowired
    private UserMapper userMapper;

    @GetMapping("/user/get")
    public User get() {
        return userMapper.selectById(1);
    }

    @PostMapping("/user/add")
    public String add() {
        User user = new User();
        user.setUname("张三丰");
        userMapper.insert(user);
        return "ok";
    }
}

// ===============================================================
@TableName("t_user")
@Data
public class User {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String uname;
}

// ===============================================================
@Mapper
public interface UserMapper extends BaseMapper<User> {
}

// ===============================================================
@SpringBootApplication(exclude = {H2ConsoleAutoConfiguration.class})
@MapperScan("com.example.shardingsphere.mapper")
public class ShardingsphereApplication {

    public static void main(String[] args) {
        SpringApplication.run(ShardingsphereApplication.class, args);
    }

}

```

- 结果展示

```txt
Logic SQL: SELECT id,uname FROM t_user WHERE id=?
SQLStatement: MySQLSelectStatement(table=Optional.empty, limit=Optional.empty, lock=Optional.empty, window=Optional.empty)
Actual SQL: slave1 ::: SELECT id,uname FROM t_user WHERE id=? ::: [1]
Logic SQL: INSERT INTO t_user  ( uname )  VALUES (  ?  )
SQLStatement: MySQLInsertStatement(setAssignment=Optional.empty, onDuplicateKeyColumns=Optional.empty)
Actual SQL: master ::: INSERT INTO t_user  ( uname )  VALUES (  ?  ) ::: [张三丰]
```

#### 2.2 垂直分片




### 3. ShardingSphere-Proxy

ShardingSphere-Proxy 是 Apache ShardingSphere 的第二个产品，它定位为透明化的数据库代理端，提供封装了数据库二进制协议的服务端版本，用于完成对异构语言的支持。 目前提供 MySQL 和 PostgreSQL 版本，它可以使用任何兼容 MySQL/PostgreSQL 协议的访问客户端（如：MySQL Command Client, MySQL Workbench, Navicat 等）操作数据，对 DBA 更加友好。


![ShardingSphere-Proxy架构](http://fanrencli.cn/fanrencli.cn/shardingsphere2.png)

#### 3.1 数据分片