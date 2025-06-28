---
title: Kafka
date: 2025-06-28 17:00:00
cover: true
top: true
categories:
  - MQ
tags:
  - Java
author: Fanrencli
---

## Kafka

### 1. 概述

Kafka 是一个分布式流处理平台，它主要用于构建实时的数据管道和流应用程序。Kafka 具有高吞吐量、可扩展性、容错性等特点，可以处理大量实时数据，并且支持多种数据源和消费方式。kafka的主要运用场景：日志收集、消息系统、用户行为追踪、流式处理等。

传统的消息队列应用场景：解耦、削峰，异步通信，缓存等。


### 2. Kafka相关组件

Kafka 的主要组件包括：生产者（Producer）、消费者（Consumer）、主题（Topic）、分区（Partition）和代理（Broker）。

![KAFKA](http://fanrencli.cn/fanrencli.cn/kafka1.png)


- 代理broker：
  - 每一个broker就是一个kafka服务器，一个集群由多个broker组成，一个broker可以保存多个分区，不同的broker之间可以配置不同的分区，这样就可以实现负载均衡。同时不同的broker可以互相进行分区备份。
  - 配置文件：
    ```txt
    # kafka的broker id,区分不同的broker
    broker.id=0
    # zk的链接地址
    zookper.connect=localhost:2181/kafka
    # kafka的监听地址
    listeners=PLAINTEXT://:9092
    # 存储日志的路径
    log.dirs=/tmp/kafka-logs
    # 日志保留天数
    log.retention.hours=168
    log.segment.bytes=1073741824
    ```
  - 启动命令
    ```sh
    bin/kafka-server-start.sh [-daemon后台服务启动] ../../config/server.properties
    ```
- 主题：
  - 主题是 Kafka 中消息的分类，每个主题可以分为多个分区。生产者将消息发送到主题，消费者主动从主题中读取消息。主题的主要作用是组织和管理消息，使得消息能够被不同的消费者处理。
    ```sh
    kafka-topics.bat --create --replication-factor 1 --partitions 1 --topic test --bootstrap-server localhost:9092 [--zookeeper localhost:2181]
    # 创建主题
    -- create
    # 删除主题
    --delete
    # 修改主题
    -- alter
    # 查看主题
    kafka-topics.bat --list --bootstrap-server localhost:9092
    # 查看主题详情
    kafka-topics.bat --describe --topic test --bootstrap-server localhost:9092
    # 设置分区数
    --partiton 3
    # 设置副本数
    --replication-factor 3
    # 更新系统默认配置
    --config x=y
    ```
- 生产者：
  - 生产者是将数据发送到 Kafka 的客户端。生产者将数据发送到指定的主题，并选择将数据发送到哪个分区。生产者还可以指定消息的键（Key）和值（Value），以便在消费者端进行消息的排序和过滤。
    ```sh
    kafka-console-producer.bat --bootstrap-server localhost:9092 --topic test
    > this is test message
    ```
- 消费者：
  - 消费者是从 Kafka 中读取数据的客户端。消费者订阅一个或多个主题，并从主题中读取数据。消费者可以通过位移（Offset）来管理读取数据的进度，以便在发生故障时可以从上次读取的位置继续读取数据。
  - 消费者实际指的是一个进程，按照消费者组（Consumer Group）进行管理，同一个消费者组内的消费者只能消费不同的分区，同一个分区只能被同一个消费者组内的消费者消费。
    ```sh
    bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic test --from-beginning[从头开始消费，否则从最新开始消费]
    ```

![KAFKA](http://fanrencli.cn/fanrencli.cn/kafka3.png)

- 分区：
  - 由于主题的数据量可能过多，因此可以将同一个主题的数据分为多个分区，不同的分区保存不同的数据，按照一定的算法进行均分。
  - 每个分区是一个有序的、不可变的消息序列。每个分区都有一个唯一的分区编号，分区编号从0开始。分区的主要作用是提高 Kafka 的吞吐量和并行处理能力。
  -  由于是依赖zk所以分区与分区之间的角色是不同的，存在一个领导者和多个追随者，领导者的角色是负责协调，追随者负责备份。消费者消费消息只针对领导者分区，追随者分区只是备份。当领导者宕机之后，追随者会选举出一个新的领导者。

### 3. Kafka生产者原理

![KAFKA](http://fanrencli.cn/fanrencli.cn/kafka2.png)

生产者主要流程：
- 生产者发送数据首先创建一个ProducerRecord对象，包含topic、key、value、timestamp等信息。经过拦截器，序列化器，分区器，经过网络传输发送到broker。
- 其中分区器的作用是根据key和partition的值来决定消息发送到哪个分区，如果key和partition都为空，则使用轮询的方式发送到不同的分区。首先将数据放入缓冲区中，然后根据batch.size和linger.ms的值来决定是否发送数据，如果缓冲区满了或者等待时间到了，则将缓冲区的数据发送到broker。
- sender线程读取数据，NetWorkClient发送请求通过Selector进行网络传输，回调函数处理broker的响应。如果broker没有应答，则sender线程会进行重试，如果重试次数超过了设定的值，则将消息放入死信队列中。
- broker接收到消息之后，首先会进行消息的校验，然后根据消息的key进行hash，根据hash值找到对应的分区，将消息存储到分区中。分区存储消息的时候，会根据消息的timestamp进行排序，如果timestamp相同，则根据消息的key进行排序，如果key相同，则根据消息的offset进行排序。

 

### kafka与rabbitmq区别

​Kafka​
- 分布式日志系统​：以分区（Partition）形式持久化消息到磁盘，依赖顺序追加写入实现高吞吐。
- 发布-订阅模型​：主题（Topic）可被多个消费者组订阅，每个分区仅由同组内一个消费者消费。
- 依赖组件​：需 ZooKeeper 协调集群（新版本逐步移除）。
   
​RabbitMQ​
- ​消息代理​：基于 AMQP 协议，核心组件为交换机（Exchange）、队列（Queue）、绑定（Binding），支持复路由规则。
- 多消息模式​：点对点、发布/订阅、工作队列等，发布订阅模式和kafka不同，因为rabbitmq的组件和kafka不同，rabbitmq是路由+队列，生产者只需要知道路由，路由按照一定的规则绑定队列，消费者订阅队列消费消息。
- ​依赖组件​：Erlang 运行时环境实现高并发。

#### 消息处理机制​

​生产与消费​

- ​Kafka​：生产者指定分区写入；消费者主动拉取（Pull），通过位移（Offset）管理进度
- ​RabbitMQ​：交换机路由消息至队列；消息推送给消费者（Push），需确认（ACK）保证可靠性

​消息顺序性​
- ​Kafka​：​分区内严格有序，分区间无序
- ​RabbitMQ​：单队列有序，但多消费者或重试可能导致乱序

​持久化​
- ​Kafka​：​所有消息默认持久化到磁盘，支持长期保留，消息消费之后并不会删除（默认保留7天），可以被多个消费组（同一个消费组只能由一个消费者消费此消息）重复消费。
- ​RabbitMQ​：需显式声明持久队列和消息，否则存内存，消息被消费之后就会删除。

