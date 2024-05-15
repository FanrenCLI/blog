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

#### AOF

#### RDB

- RDB即Redis Database,Redis数据库文件，通常为当前redis数据库的快照文件，可以用于快速恢复数据。
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