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
# 如果不存在就设置创建数据
setnx key value
# 先查后改
getset key value
```

### List

### String
### String
### String