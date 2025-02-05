---
title: 八股文
date: 2024-12-22 19:43:56
categories:
    - Java
---

# Java 后端开发面试复习提纲

## Java 基础
1. **Java 核心语法**
   - 数据类型、运算符、流程控制
   - 面向对象编程：封装、继承、多态
   - 抽象类与接口的区别
   - 重载（Overload）与重写（Override）
   - final、static、transient、volatile 关键字

2. **集合框架**
   - List、Set、Map 的区别与实现类（ArrayList、LinkedList、HashSet、TreeSet、HashMap、TreeMap）
   - HashMap 的实现原理（哈希冲突、扩容机制）
   - ConcurrentHashMap 的实现原理
   - 迭代器与 fail-fast 机制

3. **异常处理**
   - 异常分类：Checked Exception 和 Unchecked Exception
   - try-catch-finally 的执行顺序
   - 自定义异常

4. **泛型**
   - 泛型的作用与使用场景
   - 类型擦除与泛型限制

5. **反射**
   - 反射的作用与使用场景
   - Class 类、Method 类、Field 类的使用

6. **IO 与 NIO**
   - 字节流与字符流的区别
   - NIO 的核心组件：Buffer、Channel、Selector

7. **多线程**
   - 线程的创建方式（继承 Thread、实现 Runnable、Callable）
   - 线程的生命周期
   - 线程同步：synchronized、Lock、ReentrantLock
   - 线程池：ThreadPoolExecutor 参数与工作原理
   - 并发工具类：CountDownLatch、CyclicBarrier、Semaphore
   - 原子类：AtomicInteger、AtomicReference
   - volatile 关键字与内存可见性

8. **JVM**
   - JVM 内存模型：堆、栈、方法区、本地方法栈、程序计数器
   - 垃圾回收机制：GC 算法（标记清除、复制、标记整理）、GC 收集器（CMS、G1）
   - 类加载机制：双亲委派模型
   - JVM 调优：常见参数与工具（jstat、jmap、jstack）

---

## 数据库
1. **SQL 基础**
   - 增删改查（CRUD）
   - 连接查询（内连接、左连接、右连接）
   - 聚合函数与分组查询
   - 索引的作用与优化

2. **MySQL**
   - 存储引擎：InnoDB 与 MyISAM 的区别
   - 事务与 ACID 特性
   - 隔离级别：读未提交、读已提交、可重复读、串行化
   - 锁机制：行锁、表锁、间隙锁
   - 索引原理：B+ 树、哈希索引
   - 慢查询优化

3. **Redis**
   - 数据类型：String、List、Set、Hash、ZSet
   - 持久化机制：RDB 和 AOF
   - 缓存穿透、缓存雪崩、缓存击穿
   - 分布式锁的实现
   - Redis 集群模式

---

## 框架
1. **Spring**
   - IOC 与 AOP 的原理
   - Bean 的生命周期
   - Spring 事务管理
   - Spring MVC 工作原理
   - Spring Boot 自动配置原理

2. **MyBatis**
   - MyBatis 的工作原理
   - #{} 和 ${} 的区别
   - 一级缓存与二级缓存

3. **Spring Cloud**
   - 微服务架构的核心组件：Eureka、Ribbon、Feign、Hystrix、Zuul
   - 服务注册与发现
   - 负载均衡与熔断机制

---

## 分布式与中间件
1. **消息队列**
   - Kafka、RabbitMQ、RocketMQ 的区别
   - 消息可靠性保证
   - 消息重复消费与幂等性

2. **分布式事务**
   - CAP 理论
   - 两阶段提交（2PC）、三阶段提交（3PC）
   - TCC、Saga、本地消息表

3. **Zookeeper**
   - Zookeeper 的应用场景
   - ZAB 协议
   - 分布式锁的实现

4. **Dubbo**
   - Dubbo 的工作原理
   - 服务暴露与引用流程
   - 负载均衡策略

---

## 网络
1. **HTTP/HTTPS**
   - HTTP 协议与状态码
   - HTTPS 的加密原理
   - RESTful API 设计规范

2. **TCP/IP**
   - TCP 三次握手与四次挥手
   - TCP 与 UDP 的区别
   - 滑动窗口与拥塞控制

3. **WebSocket**
   - WebSocket 的工作原理
   - 与 HTTP 的区别

---

## 设计模式
1. **创建型模式**
   - 单例模式、工厂模式、建造者模式

2. **结构型模式**
   - 代理模式、适配器模式、装饰器模式

3. **行为型模式**
   - 观察者模式、策略模式、责任链模式

---

## 系统设计
1. **高并发与高可用**
   - 限流、降级、熔断
   - 负载均衡：Nginx、LVS
   - 分布式 ID 生成（Snowflake 算法）

2. **缓存设计**
   - 缓存一致性
   - 缓存更新策略

3. **数据库分库分表**
   - 垂直拆分与水平拆分
   - 分库分表中间件：ShardingSphere、MyCat

---

## 工具与 DevOps
1. **版本控制**
   - Git 常用命令
   - 分支管理策略

2. **Linux**
   - 常用命令：grep、awk、sed、ps、top
   - 文件权限管理

3. **容器化**
   - Docker 的基本使用
   - Kubernetes 的核心概念

4. **CI/CD**
   - Jenkins 的使用
   - 自动化部署流程

---

## 项目经验
1. 项目背景与架构设计
2. 技术难点与解决方案
3. 性能优化经验
4. 团队协作与沟通能力