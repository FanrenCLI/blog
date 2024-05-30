---
title: Redis
date: 2024-03-24 23:13:00
categories:
  - SQL
tags:
  - Redis
author: Fanrencli
---

## Redis五大基本类型

### String

```shell
# 移动数据到0号数据库
move key 0
# 查看过期时间
ttl key
# 设置值,NX：当没有值时设置，XX:当有值时设置,get:先返回原来的值然后再重新设置值,ex:设置过期时间second,px:设置过期时间milisecond,exat:设置过期的时间以unix的系统时间准second，pxat：设置过期的时间以unix的系统时间准millisecond,keepttl:如果修改前存在过期时间则使用之前的过期时间
set key value [NX|XX] [get] [ex|px|exat|pxat] [keepttl]
# 获取值
get key
# 设置多个值,要么都成功要么都失败
mset key value [key value...]  [NX|XX] [get] [ex|px|exat|pxat] [keepttl]
# 获取多个值
mget key [key...]
#获取指定范围的值
set name lujie
# 结果为:lu
getrange name 0 2
#结果为：qinie
setrange name 0 qin
# 数值增减:必须是数字,加1，减1，加3，减3
incr key
decr key
incrby key 3
decrby key 3
# 获取值长度
strlen key
# 字符串拼接
append key xxxx
# 数据库锁:设置一把锁并设置10秒过期
setex key 10 value
# 如果不存在就设置创建数据,存在则无效
setnx key value
# 先查后改
getset key value
```

### List

```shell 
# 从左侧添加数据
lpush key value1,value2
#从右侧添加数据
rpush key value1,value2
#从左侧拿出输出
lpop key
#从右侧拿出数据
rpop key
# 从左侧输出范围2-5数据
lrange key 2 5
# 从右侧输出范围2-5数据
rrange key 2 5
# 获取列表中序号对应的数据
lindex key 0
# 获取列表的长度
llen key
# 删除N个值等于v的数据
lrem key N v
# 截图start-end之间的数据
ltrim key start end
# 将列表中的数据移动到另一个列表中
rpoplpush source target
# 给列表某个位置改为另一个值
lset key index value
#在V数据之前或之后插入K
linsert key before/after v k

```

### Hash

```shell
# 设置值:hset user01 name lujie age 27 sex nan 
hset key field value
# 获取某个属性
hget key field
#获取所有属性
hgetall key
# 修改多个属性-弃用
hmset key field value
# 获取多个属性
hmget key field1 field2...
#获取有多少个属性
hlen key
# 删除某个或多个属性
hdel key field ...
# 判断key中是否存在某个属性
hexists key field
# 获取所有的属性
hkeys key
# 获取所有属性的值
hval key
# 给整形数值加上number
hincrby key field number
# 给浮点数加上number
hincrbyfloat key field number
# 不存在则赋值，存在了就无效
hsetnx key field value
```

### set

```shell
#添加多个元素
sadd key value...
# 遍历所有元素
smembers key
# 判断是否存在某个元素
sismember key value
#删除某个元素
srem key value
# 获取元素个数
scard key
# 随机拿出N个数据
srandmember key N
#随机拿出N个数据并删除
spop key N
# 将key1中的value移动到key2中
smove key1 key2 value
# 取出在set1中的元素同时不存在set2中的数据
sdiff set1 set2
# 取set1与set2的并集
sunion set1 set2
# 取set1和set2的交集
sinter set1 set2
# 判断set1和set2有多少个交集,limit决定返回值不能大于num，一般用于保证性能
sintercard number_sets set1 set2 [limit num]
```
### Zset

```shell
#添加元素,根据分数排序
zadd key score value[score1 value1...]
#正序后反序返回序号在范围内的数据
zrange/zrevrange key start stop [withscores]
# 获取值对应的分数
zscore key value
# 获取集合中的数量
zcard key
# 删除分数为N的数据
zrem key N

```

## Redis高级

### 持久化

- AOF和RDB可以同时开启，启动时优先加载AOF数据，没有AOF则加载RDB文件。一般RDB用于备份数据库

#### AOF

- AOF持久化策略时通过缓冲区记录所有更新数据的操作，当达到一定的阈值后将缓冲区的数据写入AOF文件中，随着AOF文件的不断膨胀，Redis还会将AOF文件进行压缩，当服务器重启时会将AOF文件中的所有命令重新加载到服务其中运行。

- 缓冲区数据写入AOF文件的三种写入策略，在redis.conf配置文件中表现为`appendfsync everysec`
  - Always:每当有更新命令出现在缓存中，就立即写入AOF文件中
  - everysec(默认的写入策略)：先把命令写入缓存中，然后每隔一秒后将缓冲区数据写入AOF文件
  - no：将命令写入缓冲区后，redis服务不决定何时写入AOF文件，而是交由操作系统进行判断何时写入AOF
- Redis默认不开启AOF数据持久化，需要在redis.conf文件中手动启用：`appendonly yes`,然后配置上述的写入策略，然后保存的文件路径在redis6中与RDB的保存路径一致，redis7中根据`appenddirname xxx`配置在RDB的路径下再创建一个xxx文件夹进行保存数据，文件名称根据配置`appendfilename xxx`进行命名。注意redis6之前AOF文件只有一个完成的文件，redis7后将一个文件拆分为3个文件进行数据保存：
  - 基本文件base，用于保存最基本的数据（应该是将增量文件中的命令数据保存而来，日志重写相关）
  - 增量文件，根据数据量可能有多个增量文件，用于保存再基本文件的基础上的增量数据
  - 清单文件，由于基本文件和增量文件数量较多，为了便于管理所以新增一个清单文件
- AOF文件异常恢复命令，如果文件损坏，可以通过redis-check-aof --fix命令进行修复
- AOF缺点:由于AOF保存的是命令，因此文件会很大，且异常恢复也会较为缓慢
- AOF优点：数据一致性较高，可靠性强
- 日志重写：随着AOF文件一直变大，当超过一定的阈值时，redis会将命令重新写为恢复数据的最终日志文件，也可以通过`bgrewriteaof`命令进行手动触发
  - 重写的过程，首先启动子进程将AOF文件复制一份临时文件用于进行重写压缩，主进程将后续的写命令一边保存再内存中，一边写入旧的AOF文件中（保证可用），当子进程完成后，主进程将内存中的命令写入新的AOF文件中，然后用新的AOF文件代替旧的AOF文件。


#### RDB

- RDB即Redis Database,Redis数据库文件，通常为当前redis数据库的快照文件，可以用于快速恢复数据。
- 配置文件中`save ""`可以禁用RDB,但是仍然可以通过save/bgsave进行备份
- 优点：
  - 适合大规模的数据恢复，速度快
  - 按照业务定时备份
  - 对数据的完整性和一致性要求不高
- 缺点：
  - 由于是全量备份所以数据量会影响系统性能
  - 备份与备份之间的数据容易丢失
- 哪些操作会产生RDB？
  - 定时备份的配置，每隔一段时间都是生成RDB
  - 手动触发备份：SAVE/BGSAVE
  - Flushall
  - shutdown执行且没有开启AOF
  - 主从复制时，主节点自动触发

### 事务

- Redis单线程所以事务不存在隔离级别，无法保证原子性，如果执行到一半失败了，无法回滚，单线程再执行一个事务中不会再执行其他命令
- Redis执行事务之前如果语法检测有问题，那么不会执行，如果语法检测没有问题，但是执行过程中发生错误，那么不会回滚，错误的执行失败，成功的执行成功。
- Redis提供了watch来实现乐观锁的机制,在执行事务之前对key进行watch，在执行事务的时候就会检查这个key是否被更改过类似于CAS。当运行exec命令之后或者断开连接，所有的watch都会被取消

```shell
# multi命令相当于mysql begin,开启事务
multi
cmd 1
cmd 2
cmd3
# exec 相当于myql end,结束事务。
exec
```

- 常用命令
  - discard
  - exec
  - multi
  - unwatch
  - watch key [key ...]
- 事务执行示例
  - 正常执行

    ```shell
    multi
    xxx
    exec  
    ```
  
  - 放弃事务

  ```shell
  multi
  xxx
  discard
  ```
  - watch监控

  ```shell
  watch key1
  multi
  xxx
  exec
  ```

### 管道

针对频繁的单一命令操作可以通过管道进行处理，提高性能

- 使用方法：通过将多条命令记录到文本中，通过redis提供的--pipeline参数进行输入服务器
- 对比原生命令（mset/mget）,原生命令是原子性，管道是非原子性，原生命令一次只能执行一种类型的命令，管道可以执行不同类型的命令，原生命令只依赖服务器，管道需要服务器和客户端一起
- 对比事务：事务具有原子性，管道不具有原子性，管道是一次性发送所有命令，事务以一条一条发，直到exec发送完，事务执行时会阻塞其他命令，但是管道不会

### 发布订阅

- subscribe channel [channel...] ：订阅多个频道的消息
- publish channel message ：向chnnel频道发布消息
- psubscribe pattern [pattern...]：订阅消息使用通配符
- pubsub channels:查看所有的频道列表
- pubsub numsub [channel...] :查看某个频道有几个订阅者
- pubsub numpat :查看使用通配符的频道数量
- unsubscribe [channel...]：取消订阅


### 主从复制

-  info replication:查看当前的主从结构
- slaveof no one：断开主从连接
- slaveof ip port：成为另一个IP的从机
- replicaof ip port：成为IP的从机
- 主机挂掉，从机不会成为主机。从机第一次连接会全量复制(通过RDB复制，复制期间新增的数据会通过命令和RDB文件一起发送过去)，后续增量复制，从机挂掉之后重新连接，会从挂掉之前的offerset开始进行同步，从机可以连接从机，但是网络延迟就会增加

![主从复制步骤](http://39.106.34.39:4567/zhucong.png)