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