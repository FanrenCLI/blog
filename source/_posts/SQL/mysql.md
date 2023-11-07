---
title: MySql高级
date: 2023-7-16 14:11:00
categories:
  - SQL
tags:
  - MySql
author: Fanrencli
---
### MySql使用

#### 表字段操作
```sql
-- 修改字段
alter table table_name modify[add] [column] field_name int not null default 0;
-- 修改字段名
alter table table_name change [column] oldname newname varchar(255);
-- 修改表名
alter table tablename1 rename to tablename2;
-- 删除字段
alter table table_name drop column field_name;
```
#### 存储函数
```sql
delimiter $ 定义结束符号
create function myfuntion(user_id int)
return varchar(50)
begin
    declare out_user_id varchar(50)
    select out_user into out_user_id from table1 where user_id = user_id;
    return out_user_id;
end$
delimiter ; 定义结束符号
select myfunction("22")；
```
#### 触发器
```sql
show triggers;
drop trigger if exist trigger_name;
delimiter $ 定义结束符号
create trigger before_insert_test1_tri
before insert on test1
for each row
begin
insert into test1() values ()
end$
delimiter ; 定义结束符号
```
#### 存储过程
```sql
-- 示例
    delimiter $ 定义结束符号
    create procedure select_all_data(in name1 varchar(20),out sex1 varchar(10))
    begin
        select sex into sex1 from users where name = name1;
    end $
    delimiter ; 防止冲突，重新定义
-- 上面定义了一个存储过程 
    set @name1= "lj"
    call select_all_data(@name1,@sex1);
    select @sex1 from dual;
```

#### 外键约束

```sql
CREATE TABLE IF NOT EXISTS test1(
    id int primary key,
    `name` varchar(10) not null DEFAULT('') UNIQUE,
    sex TINYINT not null DEFAULT('1')
);
CREATE TABLE IF NOT EXISTS test1(
    id int,
    `name` varchar(10) not null DEFAULT('') UNIQUE,
    sex TINYINT not null DEFAULT('1'),
    CONSTRAINT pk_test5_id PRIMARY KEY(id)
);
-- CONSTRAINT 用于设置名称
CREATE TABLE IF NOT EXISTS test1(
    id int,
    `name` varchar(10) not null DEFAULT('') UNIQUE,
    sex TINYINT not null DEFAULT('1'),
    PRIMARY KEY(id)
);
CREATE table test2(
    id int PRIMARY key,
    relation_id int not null UNIQUE,
    CONSTRAINT uk_f_id FOREIGN key (relation_id) REFERENCES test1 (id)
)
-- 查询表的约束 主键名称永远都是primary
select * from information_schema.table_constraints
select * from information_schema.tables
where table_name = 'test2'
-- 新增逐渐
alter table test1 add primary key (id)
-- 删除主键
alter table test1 drop primary key;
```

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
-- 确定profiling是否开启
show variables like 'profiling'
-- 查看profiles
show profiles;
-- 选择具体的一条sql执行流程
show profile [参数] for query 1;
```
参数：
- ALL:查看所有开销
- BLOCK IO:磁盘IO
- CONTEXT SWITCHES:上下文切换
- CPU:CPU的开销
- IPC:显示发送和接受的开销信息
- MEMORY：显示内存的开销



#### 存储引擎

- 存储引擎innodb引擎在5.7版本分为.frm和.idb两个文件进行数据存储，8.0之后改为.idb一个文件存储。innodb存储引擎支持事务，数据库奔溃恢复，以及行级锁，列锁，但是内存要求高
- MyISAM存储引擎在5.7版本分为.frm,.MYI,.MYD三个文件存储数据，8.0之后改为.sdi,.MYI,.MYD三个文件，不支持事务，和崩溃恢复，但是针对count(*)查询速度快，访问速度快

```sql
-- 查看引擎
show engines;
-- 查看系统默认的存储引擎
show variables like "%storage_engine%";
-- 查看数据存储路径
show variables like 'datadir';
```

#### 索引
- 聚簇索引和非聚簇索引：都是B+树，区别在于数据表的行记录是否存储在叶子节点，一般聚簇索引是以主键建立的索引，非聚簇索引一般是在主键的基础上建立的二级索引
- B树和B+树：区别在于数据是否只存储在叶子节点，B树非叶子节点也存储数据
- MyISAM引擎所建立的索引都是非聚簇索引，以主键建立的索引叶子节点都是数据表的地址，用于回表操作，二级索引与Innodb相同（因此，MyISAM的数据存储文件其中索引文件和数据文件是分开的）
- Innodb和MyISAM两种引擎所建立的索引都是B+树


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
-- 删除索引
alter table table1 drop index index_name(column_name)
drop index index_name on table1;
```
- Explain分析

```sql
-- 分析sql语句（传统方式）
explain select * from table1 where id = '1';
-- JSON方式
explain format=JSON select * from table1 where id='1';
-- 运行以上的语句后，查询优化器执行的真实sql
show warnings;
```
- id:数字对应了sql语句中有几个select关键字，当然有时候mysql会对sql语句进行优化，优化后的select关键字可能有减少
- explain执行之后的记录数对应着涉及到的表的数量（包含临时表）
- Explain分析的sql执行语句只要关注三个部分：type,key,row
- type:sql语句执行的类型，通常分为：system>const>eq_ref>ref>range>index>all,
- key:sql语句执行所用到的真正的索引，区别于possible key
- row:sql语句执行所涉及到的行数，越少越好


#### 查看系统性能参数

```sql
show [session|global] status like '参数'
```
参数：
- Connections:连接数据库的次数
- Uptime:mysql服务器上线的时间
- Slow_query:慢查询的次数
- Innodb_rows_read:select查询返回的行数
- Innodb_rows_insert:执行insert插入的行数
- Innodb_rows_update:执行update更新的行数
- Innodb_rows_delete:执行delete删除的行数
- Com_select:查询操作的次数
- Com_update:更新操作的次数
- Com_insert:插入操作的次数
- Com_delete:删除操作的次数
- last_query_cost:最后一次查询的成本

#### 慢查询

```sql
-- 查看慢查询是否开启
show variables like 'slow_query_log'
-- 开启慢查询
set @@slow_query_log = on
-- 查看慢查询日志文件的存放地址
show variables like 'slow_query_log_file'
-- 修改慢查询的时间阈值
show variables like 'long_query_time
```

- 通过mysql自带的分析工具分析慢查询的日志文件：mysqldumpslow [参数] /usr/mysql/slow.log


#### Mysql调优

- 索引情况

```sql
-- 查询冗余索引（比如对于 name 字段创建了一个单列索引，有创建了一个 name 和 code 的联合索引）
select * from sys.schema_redundant_indexes;

-- 查询未使用过的索引
select * from sys.schema_unused_indexes;

-- 查询索引的使用情况
select index_name,rows_selected,rows_inserted,rows_updated,rows_deleted 
from sys.schema_index_statistics where table_schema='dbname' ;
```

- 表相关

```sql
-- 查询表的访问量
select table_schema,table_name,sum(io_read_requests+io_write_requests) as io from
sys.schema_table_statistics group by table_schema,table_name order by io desc;

-- 查询占用bufferpool较多的表
select object_schema,object_name,allocated,data
from sys.innodb_buffer_stats_by_table order by allocated limit 10;

-- 查看表的全表扫描情况
select * from sys.statements_with_full_table_scans where db='dbname';
```

- 语句相关

```sql
-- 监控SQL执行的频率
select db,exec_count,query from sys.statement_analysis
order by exec_count desc;

-- 监控使用了排序的SQL
select db,exec_count,first_seen,last_seen,query
from sys.statements_with_sorting limit 1;

-- 监控使用了临时表或者磁盘临时表的SQL
select db,exec_count,tmp_tables,tmp_disk_tables,query
from sys.statement_analysis where tmp_tables>0 or tmp_disk_tables >0
order by (tmp_tables+tmp_disk_tables) desc

```

- IO相关

```sql
-- 查看消耗磁盘IO的文件
select file,avg_read,avg_write,avg_read+avg_write as avg_io
from sys.io_global_by_file_by_bytes order by avg_read limit 10;
```

- Innodb相关

```sql

-- 行锁阻塞
select * from sys.innodb_lock_waits;

```

- 示例：假如有（id,name）联合索引，何为索引下推？

```sql

-- 是否开启索引下推
set optimizer_switch = 'index_condition_pushdown=on';

-- 按照最左前缀原则只能使用到id索引，如果没有索引下推，则回表找到所有符合id like '3%'的数据的name列，然后一一对比是否符合'lujie'，找到所有符合的数据之后再次回表获取所有的列返回结果
-- 索引下推的存在，知道索引中包含name字段，所以会直接在索引中进行比对，找到最终的符合的结果然后回表找到所有的结果返回，减少了一次回表的时间，从某种意义上来讲也是用到了联合索引的全部字段，虽然是另一种形式。
select * from stu where id like '3%' and name ='lujie'

```

- 何为索引覆盖?
```sql
-- 当我们使用二级索引查询数据的时候，如果返回的列在二级索引中都包含则不需要进行回表取数据，二级索引中默认含有主键
select id,name from table1;

```

- 索引失效的几大情况：
    - 不符合最左前缀原则
    - 在语句中使用了计算/函数/类型自动转换
    - 使用了`range`|`!=`|`<>`|`is not null`|`like "%xxx"`
    - OR前后存在非索引的列
    - 数据库的字符集不匹配

#### 事务