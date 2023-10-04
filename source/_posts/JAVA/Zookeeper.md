---
title: Zookeeper
date: 2023-09-23 14:36:00
top: true
cover: true
categories:
  - JAVA
tags:
  - Zookeeper
author: Fanrencli
---

## Zookeeper简介

- ZooKeeper是一个分布式的，开放源码的分布式应用程序协调服务。它是一个为分布式应用提供一致性服务的软件，提供的功能包括：配置维护、域名服务、分布式同步、组服务等。
- 可以实现负载均衡，注册中心，分布式锁等功能

### Zookeeper安装配置解读

zoo.cfg文件中包含以下相关重要配置信息

- tickTime:通信心跳事件，zk服务器与客户端之间的通信间隔，或者是zk服务集群之间的通信间隔
- initLimit:初次连接不能超过的时间，
- syncLimit:主从同步时不能超过的时间，超过就认为死亡
- dataDir:zk数据存储的目录（一般为zkData）
- clientPort：zk的端口

zk的集群配置，
- 在zk安装目录下创建zkData目录，并在此目录下新建myid文件，在文件中用数字标明此服务的id
- 需要在每个zk服务的zoo.cfg文件目录下都记录整个集群的信息
```xml
<!-- 其中A为各个服务器的myid,B为各个服务器的IP,C为主机与从机交互的端口，D为选举的端口 -->
server.A=B:C:D
server.1=192.168.18.87:2888:3888
server.2=192.168.18.86:2888:3888
server.3=192.168.18.85:2888:3888
```

### Zookeeper选举操作

zk第一次的选举过程较为简单，首先在zoo.cfg文件中已经明确知道有几台服务器，某台服务器得到超过半数的投票即成为主机，第一次选举时，每个服务都会投自己一票，但是如果发现有比自己myid大的服务器就会放弃投自己的票，转而投给myid大的服务。当选举出来主机之后，再上线的服务就会直接把票给主机。此外，还需要明确三个变量。如果不是第一次选举，则按照规则：epoch大的胜出；epoch相同时，zxid大的胜出；zxid相同时，myid大的胜出。

- SID：服务器的ID，即myid文件中的数字
- ZXID：客户端对服务器发起一些操作时，回更新这个变量，集群中的所有zk服务可能不一定完全一样，有的回不一样，因此，当出现服务宕机时，会优先考虑ZXID大的服务作为主机
- EPOCH：如果集群中已经有leader，每投完一次票，leader的EPOCH就会+1


### Zookeeper命令行操作

```shell
# 启动ZK
# 在ZK的目录bin下运行zk.sh
zkServer.sh start
# 查看服务状态
zkServer.sh status
# 停止服务
zkServer.sh stop
# 重启服务
zkServer.sh restart
# 启动服务后，可以通过以下命令链接zk
zkServer.sh --server 127.0.0.1:2181

# 链接上服务器之后进行服务内部，可以通过以下命令进行操作
# zk服务类似与linux系统，数据存储的方式也是节点方式,通过以下命令查询当前目录下的数据
ls /node -s[附件次级的信息] -w[监听此节点数量的变化]
# 获取节点数据
get /node  -s[附件次级的信息] -w[监听此节点数据的变化]
# 创建节点
create /node "节点的数据" -s[含有序列] -e[临时节点，会因为重启而删除]
# 设置节点的数据
set /node "节点的数据"
# 查看节点的状态
stat /node
# 删除节点
delete /node
# 递归删除节点
deleteall /node

```

### 节点信息

- czxid:创建节点的事务ID
- ctime：节点被创建的时间
- mzxid：最后更新的事务zxid
- mtime：节点最后更新的时间
- pzxid：最后更新的子节点的id
- cversion：节点的子节点变化号，就是子节点变化的次数
- dataversion：节点数据的变化号
- aclversion：访问控制列表的变化号
- ephemeralOwner：临时检点的拥有者session，如果不是临时节点就是0
- datalength：节点数据的长度
- numChildern：节点拥有子节点的数量

### zk服务监听原理

在客户端创建连接到zk服务的过程中，会创建两个线程，一个线程用于监听zk的服务变化，一个线程用于与zk连接。当客户端需要监听zk服务中某些事件的时候，连接线程会将监听事件注册到zk服务的监听列表中，如果zk服务发生了变化，则会调用监听列表中的方法将消息发送给客户端，此时客户端会通过监听线程进行操作。注意，连接线程注册一次监听事件就只会监听一次，如果监听事件被触发了之后就不会再监听。所以如果需要持续监听则在监听的回调函数中再次监听即可。


### 代码实战

```java

```