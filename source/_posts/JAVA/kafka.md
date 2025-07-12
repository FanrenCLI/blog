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

![KAFKA-生产者](http://fanrencli.cn/fanrencli.cn/kafka2.png)

生产者主要流程：
- 生产者发送数据首先创建一个ProducerRecord对象，包含topic、key、value、timestamp等信息。经过拦截器，序列化器，分区器，经过网络传输发送到broker。
- 其中分区器的作用是根据key和partition的值来决定消息发送到哪个分区，如果key和partition都为空，则使用轮询的方式发送到不同的分区。首先将数据放入缓冲区中，然后根据batch.size和linger.ms的值来决定是否发送数据，如果缓冲区满了或者等待时间到了，则将缓冲区的数据发送到broker。
- sender线程读取数据，NetWorkClient发送请求通过Selector进行网络传输，回调函数处理broker的响应。如果broker没有应答，则sender线程会进行重试，如果重试次数超过了设定的值，则将消息放入死信队列中。
- broker接收到消息之后，首先会进行消息的校验，分区策略选择分区，将消息存储到分区中。分区存储消息的时候，会根据消息的timestamp进行排序，如果timestamp相同，则根据消息的key进行排序，如果key相同，则根据消息的offset进行排序。

#### 代码实战

```java
package com.example.kafka;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import java.util.Properties;
import java.util.concurrent.ExecutionException;
public class kafkaDemo {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        Properties properties = new Properties();
        properties.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG,"localhost:9092");
        properties.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,"org.apache.kafka.common.serialization.StringSerializer");
        properties.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,"org.apache.kafka.common.serialization.StringSerializer");
        KafkaProducer<Object, Object> kafkaProducer = new KafkaProducer<>(properties);
        for (int i = 0; i < 10; i++) {
            // 异步发送
            kafkaProducer.send(new ProducerRecord<>("test","testMessage"+i), (recordMetadata, e) -> {
                if (e == null){
                    System.out.println(recordMetadata.topic()+"--->"+recordMetadata.partition());
                }
            });
        }
        // 将缓冲区的数据刷掉
        kafkaProducer.flush();
        Thread.sleep(5000);
        for (int i = 0; i < 10; i++) {
            // 同步发送
            kafkaProducer.send(new ProducerRecord<>("test","testMessage123"+i)).get();
        }
        // 关闭资源
        kafkaProducer.close();
    }
}
```

#### 生产者分区原理

1. 分区策略
- 生产者发送消息时，如果选择了分区，那么就发送到对应的分区中
- 如果不选择那么会根据key和partition的hash%numberofPartition值来决定消息发送到哪个分区，
- 如果key和partition都为空，采用黏性分区器，第一次随机选择一个分区，后续发送到这个分区，直到这个分区满了，再随机选择一个分区，以此类推。

2. 自定义分区器

如果实际开发过程中，分区策略不满足我们的需求，那么我们可以定义自己的分区器

```java
package com.example.kafka;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.Partitioner;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.Cluster;

import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ExecutionException;
public class kafkaDemo {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        Properties properties = new Properties();
        properties.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG,"localhost:9092");
        properties.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,"org.apache.kafka.common.serialization.StringSerializer");
        properties.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,"org.apache.kafka.common.serialization.StringSerializer");
        // 关联自定义分区器
        properties.put(ProducerConfig.PARTITIONER_CLASS_CONFIG, myPartition.class.getName());
        KafkaProducer<Object, Object> kafkaProducer = new KafkaProducer<>(properties);
        for (int i = 0; i < 10; i++) {
            // 异步发送
            kafkaProducer.send(new ProducerRecord<>("test","testMessage"+i), (recordMetadata, e) -> {
                if (e == null){
                    System.out.println(recordMetadata.topic()+"--->"+recordMetadata.partition());
                }
            });
        }
        // 将缓冲区的数据刷掉
        kafkaProducer.flush();
        Thread.sleep(5000);
        for (int i = 0; i < 10; i++) {
            // 同步发送
            kafkaProducer.send(new ProducerRecord<>("test","testMessage123"+i)).get();
        }
        // 关闭资源
        kafkaProducer.close();
    }
    // 自定义分区器
    class myPartition implements Partitioner {
        @Override
        public int partition(String topic, Object key, byte[] keyBytes, Object value, byte[] valueBytes, Cluster cluster) {

            String msgValue = value.toString();

            if (msgValue.contains("test")){
                return 0;
            }else{
                return 1;
            }
        }

        @Override
        public void close() {

        }

        @Override
        public void configure(Map<String, ?> configs) {

        }
    } 
}

```

#### 生产者的吞吐量

1. 批次大小 batch.size
- 生产者发送消息时，会批量发送，每次发送的批次大小默认为16k，如果批次满了，那么就会发送消息。
2. 延迟时间 linger.ms
- 如果批次没有满，那么会等待linger.ms时间，如果时间到了，就会发送消息，如果批次还是没有满，那么也会发送消息。
3. 压缩算法 compression.type
- 生产者发送消息时，可以压缩消息，压缩算法有：none、gzip、snappy、lz4、zstd，默认为none，不压缩，压缩可以减少网络传输的数据量，但是会增加cpu的消耗。

```java
package com.example.kafka;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import java.util.Properties;
import java.util.concurrent.ExecutionException;
public class kafkaDemo {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        Properties properties = new Properties();
        properties.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG,"localhost:9092");
        // 序列化
        properties.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,"org.apache.kafka.common.serialization.StringSerializer");
        properties.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,"org.apache.kafka.common.serialization.StringSerializer");
        // 缓冲区大小
        properties.put(ProducerConfig.BUFFER_MEMORY_CONFIG,"33554432");
        // 批次大小
        properties.put(ProducerConfig.BATCH_SIZE_CONFIG,"16384");
        // 延迟时间
        properties.put(ProducerConfig.LINGER_MS_CONFIG,"1");
        // 压缩算法
        properties.put(ProducerConfig.COMPRESSION_TYPE_CONFIG,"snappy");
        KafkaProducer<Object, Object> kafkaProducer = new KafkaProducer<>(properties);
        for (int i = 0; i < 10; i++) {
            // 异步发送
            kafkaProducer.send(new ProducerRecord<>("test","testMessage"+i), (recordMetadata, e) -> {
                if (e == null){
                    System.out.println(recordMetadata.topic()+"--->"+recordMetadata.partition());
                }
            });
        }
        // 将缓冲区的数据刷掉
        kafkaProducer.flush();
        Thread.sleep(5000);
        for (int i = 0; i < 10; i++) {
            // 同步发送
            kafkaProducer.send(new ProducerRecord<>("test","testMessage123"+i)).get();
        }
        // 关闭资源
        kafkaProducer.close();
    }
}

```

#### 数据的可靠性

1. acks

- acks=0：生产者在消息发送出去之后，不需要等待任何的响应，所以这个方式会有最大的吞吐量，但是也是最不可靠的。
- acks=1：生产者在消息发送出去之后，只要leader副本收到消息，就会响应，所以这个方式会有一定的吞吐量，但是也是比较可靠的。
- acks=all：生产者在消息发送出去之后，只有当ISR列表中所有的副本都收到消息，才会响应，所以这个方式会有最小的吞吐量，但是也是最可靠的。如果某个副本长时间（默认30s）未向leader发送请求同步数据，那么leader会将该副本从ISR列表中移除，当ISR列表为空时，leader会变为follower，此时生产者发送的消息会失败。
 
通常在生产环境中，如果时日志场景，通常可以设置为1，如果是重要数据则设置all，同时保证副本大于等于2，以及应答副本大于等于2。在某些场景中，如果leader和副本都收到了消息，但是在回应前宕机了，此时生产者重复发送，可能导致数据重复。

```java
// 应答策略
properties.put(ProducerConfig.ACKS_CONFIG,"all");
// 重试次数默认int的最大值
properties.put(ProducerConfig.RETRIES_CONFIG,3);
```

#### 数据重复

根据数据可靠性原理的讲解，在某一些场景中可能存在重复数据的场景，此时我们引入幂等性问题，当我们结合数据的可靠性和幂等性时，那么就可以精确的传输数据，保证数据不丢失，不重复。

1. 幂等性： 指发送方重复发送消息，broker只会持久化一次，在0.11版本后，kafka默认提供了幂等性，只需要将producer的配置中enable.idempotence设置为true即可。原理：通过三元组 `<PID, Partition, SeqNumber>`,PID表示一个会话，Partition表示分区，SeqNumber表示序列号，broker会为每个生产者分配一个PID，生产者发送消息时，会携带PID和序列号，broker接收到消息后，会判断该消息是否重复，如果重复，则不持久化。所以，只能保证单分区单会话不重复。
  

2. 事务：幂等性在保证数据的重复性只保证单分区单会话不重复，但是生产者重启之后还会存在问题，因为要解决这个问题，还需要使用kafka的事务特性,要使用事务特性，首先需要开启幂等性，当然幂等性是默认开启的。通过指定唯一的事务id，broker会保存事务id和PID的对应关系，当生产者发送消息时，会携带事务id，broker接收到消息后，会判断该消息是否重复，如果重复，则不持久化。同时，事务id可以保证多个生产者同时发送消息，不会产生冲突。

用户需为生产者配置全局唯一且持久化的事务ID​（如transactional.id=order_service）。该ID在生产者重启后保持不变，作为跨会话的锚点。首次初始化事务时，事务协调器（Transaction Coordinator）将事务ID与PID绑定，并持久化到内部Topic（__transaction_state）。生产者重启后，通过相同事务ID向协调器请求，即可恢复原始PID，而非生成新PID

```java
package com.example.kafka;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import java.util.Properties;
public class kafkaDemo {
    public static void main(String[] args)  {
        Properties properties = new Properties();
        properties.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG,"localhost:9092");
        // 序列化
        properties.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,"org.apache.kafka.common.serialization.StringSerializer");
        properties.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,"org.apache.kafka.common.serialization.StringSerializer");
        // 指定事务id
        properties.put(ProducerConfig.TRANSACTIONAL_ID_CONFIG,"transactionalId_01");
        KafkaProducer<Object, Object> kafkaProducer = new KafkaProducer<>(properties);
        kafkaProducer.initTransactions();
        kafkaProducer.beginTransaction();
        try{
            for (int i = 0; i < 10; i++) {
                // 异步发送
                kafkaProducer.send(new ProducerRecord<>("test","testMessage"+i));
            }
            kafkaProducer.commitTransaction();
        }catch (Exception e){
            kafkaProducer.abortTransaction();
        }finally {
            kafkaProducer.close();
        }
    }
}

```

#### 数据有序性

1. 单分区有序：生产者发送消息时，会按照顺序发送，消费者按照顺序消费，但是生产者发送消息时，如果发送到不同的分区，那么消息的顺序就无法保证了，因为不同的分区之间无法保证顺序。如果未开启幂等，要保证单分区有序，需要将max.in.flight.requests.per.connection设置为1，但是这样会降低吞吐量，所以通常情况下，我们开启幂等性，这样max.in.flight.requests.per.connection默认为5，这样也可以保证单分区有序，因为幂等性传输存在序列号，可以进行排序。如果大于5，因为缓存只能保存5个消息，超过溢出导致无法排序。

### 4. broker

#### broker数据迁移

broker上线新节点和下线旧节点时，需要进行数据迁移操作，避免影响集群性能。数据迁移过程如下：
1. 将数据从原来的节点重新负载均衡到其他所有节点上
   - 生成迁移计划,创建JSON文件`topics-to-move.json`：  
    ```json
    {"topics": [{"topic": "test_topic1"}, {"topic": "test_topic2"}], "version": 1}
    ```
   - 执行命令生成迁移计划：
    ```bash
    bin/kafka-reassign-partitions.sh --zookeeper zk:2181 --topics-to-move-json-file topics-to-move.json   --broker-list "4,5,6" --generate
    ```
   - 输出结果包含当前分区分布和建议的新分布,将建议的JSON保存为reassignment.json：
    ```bash
    bin/kafka-reassign-partitions.sh --zookeeper zk:2181  --reassignment-json-file reassignment.json --execute
    ```
2. 将数据从某个节点迁移到其他节点上，从而下线,同理

#### broker副本信息

- kafka副本的作用：提高数据的可靠性，提高数据的可用性，因为副本可以提供数据的读取，当某个节点宕机时，可以从副本中读取数据。
- kafka默认副本1个，生产环境至少2个，保证副本的可靠性。太多的副本会占用大量的存储空间。
- kafka的副本分为Leader和Follower，Leader负责读写数据，Follower负责同步数据，当Leader宕机时，Follower会选举出一个新的Leader。
- kafka分区中的所有副本统称为AR（Assigned Replicas）
  - AR = ISR + OSR
  - ISR：In-Sync Replicas，与Leader保持同步的Follower集合，只有ISR中的副本才会被选举为Leader，如果follower长时间未向leader发送请求同步数据，那么leader会将该副本从ISR列表中移除，时间阈值为`replica.lag.time.max.ms`,默认30s。
  - OSR：Out-of-Sync Replicas，与Leader同步滞后过多的Follower集合。

![KAFKA副本Leader选举过程](http://fanrencli.cn/fanrencli.cn/kafka5.png)

- 选举过程（每个分区都有一个leader和follower）：
  - 首先broker启动后会在zookeeper中创建临时节点`/controller`，该节点保存了当前集群的controller信息。
  - 每个broker都有controller，谁先注册到zookeeper中，谁就是controller，controller会监听zookeeper中的`/controller`节点，如果controller宕机，跟随者会监听到该节点消失，然后重新竞争controller节点。
  - 确定了controller之后，由controller将当前分区的所有ISR集合信息上传展zk，其他controlelr同步信息。总的来说，controller就是竞争zk中的临时节点，谁抢到谁就控制选举，并将所有的ISR信息上传zk，其他controller同步信息。
  - 当分区leader宕机后，controller会监听到该分区leader消失，然后从zk中获取ISR集合，然后从ISR集合中选举新的leader，选举规则如下：确认ISR存活的节点信息，然后根据AR中的排序，选择靠前的节点作为leader

- LEO：Log End Offset，每个副本的最后一个offset
- HW：High Watermark，消费者可见的offset，HW是LEO中的最小值，消费者只能消费到HW之前的消息，HW之前的消息是可用的，HW之后的消息是未同步的，消费者不能消费到HW之后的消息。
- 当leader收到生产者发送的数据之后，更新leo，然后获取所有follower的leo数据，然后更新自己的HW。消费者拉拉取数据后，更新leo，并根据leader的HW更新自己的HW。但是leader的HW还是0，因此follower还是0，只有等下次leader更新HW之后，follower才会更新HW。

#### 文件存储

- 分区文件：每个topic的文件以`topic_name-partition_id`命名，每个分区文件中包含多个segment文件，每个segment文件包含多个log文件和index文件，log文件存储数据，index文件存储索引，索引文件中存储的是offset和position，position是log文件中的偏移量，offset是消息的偏移量，通过offset可以快速定位到消息的位置。index是稀疏索引，大约每往log文件中写入4kb的数据，就会在index文件中写入一个索引，因此index文件的大小会比log文件小很多，但是索引的查找速度会比log文件快很多。
- 文件清除策略：`log.retention.hours` 默认7天， `log.retention.check.interval.ms`定时检查是否过期的时间间隔。`log.cleanup.policy`默认是delete，如果设置为compact，那么会进行压缩，将相同key的消息合并，只保留最新的消息，这样就可以减少存储空间，但是会降低写入速度，因为需要合并消息。文件的过期时间判断是根据log文件中的最后一条消息的时间戳来判断的，如果最后一条消息的时间戳超过了过期时间，那么整个log文件都会被删除。

#### 高效读写

1. kafka本身是分布式集群，可以采用分区技术，将数据分散到不同的分区中，这样就可以实现数据的并行读写，提高读写速度。
2. 读取数据采用稀疏索引，可以快速定位数据
3. 顺序读写：kafka生产者产生的数据写入log文件采用追加写的方式，这样就可以避免磁盘的随机读写，提高读写速度。
4. 页缓存+零拷贝：kafka写入数据时，会将数据写入页缓存，然后由操作系统将数据写入磁盘，这样就避免了数据的拷贝，提高了写入速度。读取数据时，也是从页缓存中读取，如果缓存中没有则从文件中读取，读取到数据之后不会切换用户态进行数据发送，而是直接从缓存中将数据发送给消费者。


### 5. 消费者

消息消费的模式一般分为两种：pull（拉取）和push（推送）。
- pull：消费者主动从broker拉取数据，消费者可以根据自己的消费能力来拉取数据，可以控制消费速度，但是需要消费者自己处理拉取数据的时间间隔。如果kafka没有数据，消费者会陷入空循环
- push：broker主动将数据推送给消费者，broker可以根据自己的能力来推送数据，但是消费者无法控制消费速度，可能会导致消费者处理不过来，导致消息堆积。

#### 消费者工作流程

- 一个消费者可以消费多个分区的数据
- 一个消费者组中的消费者可以消费不同分区的数据，但是一个分区只能被一个消费者组中的一个消费者消费
- 消费者消费的offset保存在kafka中(__consumer_offsets)，当消费者宕机时，可以从kafka中获取到上次消费的offset，然后继续消费,不保存在zk中是因为可以降低与zk的交互

#### 消费者组

消费者组：消费者组是由多个消费者组成的，消费者组可以消费多个分区的数据，消费者组中的消费者可以消费不同分区的数据，但是一个分区只能被一个消费者组中的一个消费者消费。
  
消费者组中的消费者消费数据时，会根据分区数和消费者数来确定每个消费者消费的分区，如果分区数小于消费者数，那么会有消费者空闲，如果分区数大于消费者数，那么会有消费者消费多个分区。

消费者组初始化流程：
  - coordinator:辅助实现消费者组的初始化和分区分配，每个消费者组都会有一个coordinator，coordinator在kafka集群中每个broker中，根据hscode(groupId)%50（__consumer_offsets的分区数量）,选择一个broker作为coordinator，同时消费者的offset也往这个分区提交。
  - 所有消费者向coordinator发送JoinGroup请求，coordinator会根据消费者的信息，选择一个消费者作为leader，其他消费者作为follower.
  - leader消费者根据分区策略`partition.assignment.strategy`：range、roundrobin、sticky、CooperativeSticky，制定分区计划，
    - range(默认):针对某个tpoic分区，将分区按照顺序分配给消费者，例如有3个分区，2个消费者，那么第一个消费者消费前两个分区，第二个消费者消费最后一个分区。
    - roundrobin:针对所有topic的分区，将分区按照顺序分配给消费者，例如有3个分区，2个消费者，那么第一个消费者消费第一个分区，第二个消费者消费第二个分区，第一个消费者消费第三个分区，然后循环。
    - sticky:数量尽可能均匀，且随机分配，不按照顺序分配，当消费者出现问题后，再分配时会尽量不变动原来消费者的分区，只在基础上进行微调
  - leader消费者制定的分区计划发送给coordinator，coordinator将分区计划发送给所有消费者。每个消费者与coordinator保持心跳，如果消费者与coordinator长时间(`session.timeout.ms`=45s)没有心跳或者消费者处理消息时间过长(`max.poll.interval.ms`=5min)，那么coordinator会将消费者从消费者组中移除并触发再平衡。


![KAFKA消费流程](http://fanrencli.cn/fanrencli.cn/kafka4.png)

消费流程：
  - 消费者发起消费请求sendFetchRequest后，会创建socket客户端，客户端从kafka获取数据，请求参数：`Fetch.min.bytes`每批次最小抓取大小1b、`Fetch.max.wait.ms`一批数据最小值未达到的超时时间500ms、`Fetch.max.bytes`每批次最大抓取大小50M
  - 拉取完成completedFetches拉取数据之后放入队列中，然后消费者从队列中获取数据`Max.poll.records`一次拉取数据返回消息的最大条数500。获取数据后还需要通过反序列化和拦截器处理。

代码示例
```java
package com.example.kafka;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.serialization.StringDeserializer;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class Consumer {
    public static void main(String[] args) {
        Properties properties = new Properties();
        properties.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        properties.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        properties.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        // 指定groupid
        properties.put(ConsumerConfig.GROUP_ID_CONFIG, "group1");
        KafkaConsumer<String,String> kafkaConsumer = new KafkaConsumer<>(properties);
        List<String> topic = new ArrayList<>();
        topic.add("test");
        kafkaConsumer.subscribe(topic);
        while(true){
            ConsumerRecords<String, String> consumerRecords = kafkaConsumer.poll(Duration.ofSeconds(1));
            for (ConsumerRecord<String, String> consumerRecord : consumerRecords) {
                System.out.println(consumerRecord);
            }
        }

    }
}

```
```java
public class Consumer {
    public static void main(String[] args) {
        Properties properties = new Properties();
        properties.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        properties.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        properties.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        properties.put(ConsumerConfig.GROUP_ID_CONFIG, "group1");
        // 消费者组分区策略
        properties.put(ConsumerConfig.PARTITION_ASSIGNMENT_STRATEGY_CONFIG, RangeAssignor.RANGE_ASSIGNOR_NAME);
        KafkaConsumer<String,String> kafkaConsumer = new KafkaConsumer<>(properties);
        // 消费指定主题的特定分区数据
        List<TopicPartition> topic = new ArrayList<>();
        topic.add(new TopicPartition("test",0));
        kafkaConsumer.assign(topic);
        while(true){
            ConsumerRecords<String, String> consumerRecords = kafkaConsumer.poll(Duration.ofSeconds(1));
            for (ConsumerRecord<String, String> consumerRecord : consumerRecords) {
                System.out.println(consumerRecord);
            }
        }

    }
}
```

#### offset

消费者再消费消息时，需要知道从哪个位置开始消费，这个位置就是offset，offset是消费者组级别的，每个消费者组都有自己的offset。offset信息保存再kafka中，每个消费者的offset信息保存于对应分区所在broker的__consumer_offsets主题中(0.9版本以前在zk)，每个消费者组对应一个分区，分区数由`offsets.topic.num.partitions`配置，默认50个分区。因为一个消费者组只能有一个消费者消费一个分区，所以offset保存时按照key-value进行保存，其中key=groupId-topic-partition，value=offset。消费者在消费消息时，会从__consumer_offsets主题中获取offset，然后从对应分区中获取消息。

- enable.auto.commit：是否自动提交offset，默认true
- auto.commit.interval.ms：自动提交offset的时间间隔，默认5s

kafka默认自动提交offset,让开发者专注于业务逻辑，但是如果需要手动提交也可以实现

- 手动提交offset
  - 同步提交：`consumer.commitSync()`，会阻塞直到提交成功，如果提交失败会抛出异常，需要捕获异常进行处理。
  - 异步提交：`consumer.commitAsync()`，不会阻塞，提交成功会回调一个接口，提交失败不会抛出异常，需要手动捕获异常进行处理。
- 指定offset消费：`auto.offset.reset`
  - earliest：从最早的消息开始消费,`--from-beginning`
  - latest：从最新的消息开始消费
  - none：如果offset不存在，则抛出异常
- 指定时间戳消费：`consumer.seek(topicPartition, timestamp)`，从指定时间戳开始消费

代码示例
```java
package com.example.kafka;

import org.apache.kafka.clients.consumer.*;
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.common.serialization.StringDeserializer;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Set;

public class Consumer {
    public static void main(String[] args) {
        Properties properties = new Properties();
        properties.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        properties.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        properties.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        properties.put(ConsumerConfig.GROUP_ID_CONFIG, "group1");
        properties.put(ConsumerConfig.PARTITION_ASSIGNMENT_STRATEGY_CONFIG, RangeAssignor.RANGE_ASSIGNOR_NAME);
        KafkaConsumer<String,String> kafkaConsumer = new KafkaConsumer<>(properties);
        // 消费指定主题的特定分区数据
        List<TopicPartition> topic = new ArrayList<>();
        topic.add(new TopicPartition("test",0));
        kafkaConsumer.assign(topic);
        // 指定消费offset,先获取对应分区的offset,由于获取时可能还未链接上对应分区，可能为空
        Set<TopicPartition> assignment = kafkaConsumer.assignment();
        while(assignment.size()==0){
            kafkaConsumer.poll(Duration.ofMillis(100));
            assignment = kafkaConsumer.assignment();
        }
        for (TopicPartition topicPartition : assignment) {
            kafkaConsumer.seek(topicPartition, 100);
        }
        while(true){
            ConsumerRecords<String, String> consumerRecords = kafkaConsumer.poll(Duration.ofSeconds(1));
            for (ConsumerRecord<String, String> consumerRecord : consumerRecords) {
                System.out.println(consumerRecord);
            }
            // 异步提交
            kafkaConsumer.commitAsync();
            // 同步提交
            kafkaConsumer.commitSync();
        }

    }
}
```

```java
package com.example.kafka;

import org.apache.kafka.clients.consumer.*;
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.common.internals.Topic;
import org.apache.kafka.common.serialization.StringDeserializer;

import java.time.Duration;
import java.util.*;

public class Consumer {
    public static void main(String[] args) {
        Properties properties = new Properties();
        properties.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        properties.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        properties.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        properties.put(ConsumerConfig.GROUP_ID_CONFIG, "group1");
        properties.put(ConsumerConfig.PARTITION_ASSIGNMENT_STRATEGY_CONFIG, RangeAssignor.RANGE_ASSIGNOR_NAME);
        KafkaConsumer<String,String> kafkaConsumer = new KafkaConsumer<>(properties);
        // 消费指定主题的特定分区数据
        List<TopicPartition> topic = new ArrayList<>();
        topic.add(new TopicPartition("test",0));
        kafkaConsumer.assign(topic);
        // 指定消费offset,先获取对应分区的offset,由于获取时可能还未链接上对应分区，可能为空
        Set<TopicPartition> assignment = kafkaConsumer.assignment();
        while(assignment.size()==0){
            kafkaConsumer.poll(Duration.ofMillis(100));
            assignment = kafkaConsumer.assignment();
        }
        // 获取所有分区对象，并设置不同分区的时间跨度一天前
        Map<TopicPartition,Long> map = new HashMap<>();
        for (TopicPartition topicPartition : assignment) {
            map.put(topicPartition,System.currentTimeMillis()-1*24*3600*1000);
        }
        // 根据接口查询得到不同分区一天前的现在的offset
        Map<TopicPartition, OffsetAndTimestamp> topicPartitionOffsetAndTimestampMap = kafkaConsumer.offsetsForTimes(map);
        // 为不同分区设置offset
        for (TopicPartition topicPartition : assignment) {
            kafkaConsumer.seek(topicPartition, topicPartitionOffsetAndTimestampMap.get(topicPartition).offset());
        }
        while(true){
            ConsumerRecords<String, String> consumerRecords = kafkaConsumer.poll(Duration.ofSeconds(1));
            for (ConsumerRecord<String, String> consumerRecord : consumerRecords) {
                System.out.println(consumerRecord);
            }
            // 异步提交
            kafkaConsumer.commitAsync();
            // 同步提交
            kafkaConsumer. commitSync();
        }

    }
}
```

- 重复消费：自动提交offset存在时间延迟，每隔5秒提交一次，如果下一次提交之前宕机，重启后可能重复消费。解决方法可以根据消息的唯一标识判断是否消费过，结合支持事务的数据库mysql进行处理。
- 消息丢失：消费者还未消费完数据，提交offset，然后宕机，重启后无法重新消费。
- 数据积压：消费者消费速度慢，生产者生产消息速度快，导致消息积压。提高分区数，增加消费者数量，提高一次性拉取的数据量

### 6. SpringBoot集成

- 配置信息

```properties
spring.application.name=springBootKafka
server.port=8023

spring.kafka.bootstrap-servers=localhost:9092

spring.kafka.producer.key-serializer=org.apache.kafka.common.serialization.StringSerializer
spring.kafka.producer.value-serializer=org.apache.kafka.common.serialization.StringSerializer


spring.kafka.consumer.key-deserializer=org.apache.kafka.common.serialization.StringDeserializer
spring.kafka.consumer.value-deserializer=org.apache.kafka.common.serialization.StringDeserializer

spring.kafka.consumer.group-id="fanren"
```

- 生产者代码

```java
package com.example.springbootkafka;


import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.annotation.KafkaListener;

@Configuration
public class ConsumerController {

    @KafkaListener(topics = "test")
    public void listen(String record) {
        System.out.println(record);
    }
}

```

- 消费者代码

```java
package com.example.springbootkafka;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ProducerController {

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    @RequestMapping("test")
    public String sendMessage(String message) {
        kafkaTemplate.send("test", message);

        return "ok";
    }
}

```

### 7. kafka调优

场景说明：100w日活，每人每天100条日志，总共每天1亿条数据，平均1150/s，峰值20000/s，数据量：20M/s




### 8. kafka与rabbitmq区别

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

