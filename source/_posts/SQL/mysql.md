---
title: MySql高级
date: 2023-7-16 14:11:00
categories:
  - SQL
tags:
  - MySql
author: Fanrencli
---

### MySql高级特性

#### 字符集

```sql
-- 查看数据库变量
show variables like "%character%";
```

#### 用户操作

```sql
-- 查看所有用户
use mysql;
SELECT host,user from user;
-- 创建用户
create user 'lujie_test'@'localhost' identified by '123qwe';
create user 'lujie_test'@'%' identified by '123qwe';
create user 'lujie_test' identified by '123qwe';
-- 修改用户
update user set user='lujie' where user='lujie_test' and host = 'localhost';
flush privileges;
-- 删除用户 (推荐使用drop)
drop user 'lujie'@'localhost';

delete from user where user='lujie' and host='localhost';
flush privileges;

-- 修改当前用户密码
alter user user() identified by 'changepwd';
alter user 'lujie'@'localhost' identified by 'changepwd1';
-- 查询当前用户所拥有的权限
show grants;
-- 授予用户权限
grant select,update on dbtest.* to 'lujie'@'localhost';
grant all privileges on *.* to 'lujie'@'localhost';
-- 回收用户全新啊
revoke select,update on dbtest.* to 'lujie'@'localhost';
revoke all privileges on *.* to 'lujie'@'localhost';
-- 创建角色
create role 'manage'@'localhost';
grant select,update on dbtest.* to 'manage';
grant all privileges on *.* to 'lujie'@'localhost';
-- 查看角色的权限
show grants for 'manage';
-- 回收权限
revoke select,update on dbtest.* to 'manage'@'localhost';
revoke all privileges on *.* to 'manage'@'localhost';
-- 删除角色
drop role 'manage';
```
#### SQL执行流程

```sql
-- 首先执行一条sql语句
select * from table1;
-- 查看profiles
show profiles;
-- 选择具体的一条sql执行流程
show profile for query 1;
```

#### 存储引擎

-- 存储引擎innodb引擎在5.7版本分为.frm和.idb两个文件进行数据存储，8.0之后改为.idb一个文件存储。innodb存储引擎支持事务，数据库奔溃恢复，以及行级锁，列锁，但是内存要求高
-- MyISAM存储引擎在5.7版本分为.frm,.MYI,.MYD三个文件存储数据，8.0之后改为.sdi,.MYI,.MYD三个文件，不支持事务，和崩溃恢复，但是针对count(*)查询速度快，访问速度快

```sql
-- 查看引擎
show engines;
-- 查看系统默认的存储引擎
show variables like "%storage_engine%";
-- 查看数据存储路径
show variables like 'datadir';
```

#### 索引
- 查看索引
```sql
show index from table1
```

- 创建普通索引
```sql
create table table1(
  id int,
  name varchar(10),
  addr varchar(10),
  [unique index|index] id_name(id)
)
-- 显示创建索引
alter table table1 add index index_name(column_name)
alter table table1 add unique index index_name(column_name1,column_name1)
create index index_name on table1(column_name);
```