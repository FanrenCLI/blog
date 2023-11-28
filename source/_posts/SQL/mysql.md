---
title: MySql高级
date: 2023-11-24 20:43:44
categories:
  - SQL
tags:
  - MySql
author: Fanrencli
---
## MySql使用

### 表字段操作
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
### 存储函数
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
### 触发器
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
### 存储过程
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

### 外键约束

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

## MySql高级特性

### 字符集

```sql
-- 查看数据库变量
show variables like "%character%";
```

### 用户操作

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
### SQL执行流程

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



### 存储引擎

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

### 索引
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


### 查看系统性能参数

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

### 慢查询

```sql
-- 查看慢查询是否开启
show variables like 'slow_query_log'
-- 开启慢查询
set @@slow_query_log = on
-- 查看慢查询日志文件的存放地址
show variables like 'slow_query_log_file'
-- 修改慢查询的时间阈值
show variables like 'long_query_time'
```

- 通过mysql自带的分析工具分析慢查询的日志文件：mysqldumpslow [参数] /usr/mysql/slow.log


### Mysql调优

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

### 事务

- 显式事务
    - 可以通过*begin*开启事务，或者通过*start transaction*开启事务，相比与*begin*,*start transaction*后面可以接着参数：`[read only|read wirte|with consistent snapshot]`,第三个参数可以第一个参数或者第二个参数配合使用

    ```sql
    start transaction read only;
    start transaction read write,with consistent snapshot;
    ```

- 隐式事务

    - auto_commit变量为on

    ```sql
    show variables like 'auto_commit'
    ```
    - DDL语句会自动提交
    - 修改表结构等操作
    - 显式开启事务时会自动提交上一个事务

- 事务并发存在的问题
    - 脏读：事务A读取了事务B中没有提交的新增/修改的数据
    - 不可重复度：事务A读取了一个数据，然后事务B进行了修改并提交，然后事务A再读取到了事务B提交的数据
    - 幻读：事务A读取了一些数据，事务B插入了一些数据并提交，事务A再同样条件读取的时候发现多了数据，如果少了则是不可重复读
    - 为什么幻读和不可重复读看起来是同一种场景，却专门分开说？是因为如果采用MVCC+加锁的方式解决读写并发的情况时，MVCC会同时解决这两种问题，但是如果当前的业务场景不允许读写并行，那么读的时候加锁就会变得复杂，因为如果只对读的数据（>0）进行加锁，那么写操作新增的数据(1,3)可能会影响读的数据，这时候就引申出来的不同的锁类型（临界锁，间隙锁等）
- 数据库四种隔离级别
    - READ UNCOMMITED：什么都没解决
    - READ COMMITED:解决脏读，MVCC在这种隔离级别中，每次select查询语句都会新建一个当前的视图READVIEW
    - REPEATABLE READ：解决脏读和不可重复读，MVCC在这种隔离级别中，只有第一次select会新建一个视图，后续都只读这个视图，因此解决了幻读问题
    - SERIALIZABLE：解决所有问题

```sql
-- 查看数据库的隔离级别
show variables like '%isolation%'
```

### 锁&MVCC
首先明确为什么会出现隔离级别的概念，是因为并发访问同一个资源的问题，如果没有并发就不需要隔离级别。因为并发的出现就需要加锁，而并发又分为三类
- 读读并发：没有任何问题
- 读写并发：这个类别最重要，MVCC就是处理这类问题的，在这种场景下就会出现之前的：脏读，不可重复读，幻读的情况。在某些场景中，如果允许读和写并行，那么读操作通过MVCC进行控制，写操作通过加锁进行控制。在某些场景中，读和写不允许并行（判断银行卡余额是否足够），那么读写都加锁(这时候加锁的情况就复杂了，也就是因为这些复杂的情况才会衍生出不可重复读和幻读)。
- 写写并发：这类只能加锁，一条记录在被写操作的时候会生成这个事务对应的锁，其他事务想要操作这个记录也需要生成锁结构，然后进行等待。这个锁结构包含两条信息，一个是表明这个锁是哪个事务，一个是表明这个锁是否需要等待。

#### 锁的分类

- 按照对数据的操作类型：共享锁（读锁）和排他锁（写锁）

```sql
-- 注意增删改操作会自动加X锁，查询操作需要显式加锁，不过查询的时候没有显式加锁，那么就会通过MVCC来保证一致性
-- 旧版本，共享锁
select ... LOCK IN SHARE MODE;
-- 8.0版本，共享锁
select ... FOR SHARE 
-- 排他锁
select ... FOR UPDATE
-- 如果获取不到就直接返回
select ... FOR UPDATE nowiat
-- 跳过有锁的记录，返回没有锁的记录
select ... FOR UPDATE SKIP LOCKS
```

- 按照锁的粒度划分：表锁，行锁，页级锁
    - 表锁：表级的S锁（共享锁）和X（排他锁）锁（这个一般为MyISAM表使用，虽然innodb也可以用但是不推荐，innodb一般使用后面的三种锁，后面的三种锁也分为S/X锁），意向锁，自增锁，MDL锁
        1. 表锁一般不再innodb中使用，因为innodb中可以使用行锁，不过非要用也可以

            ```sql
            -- 先查看表是否有锁
            show open tables where in_use>0;
            -- 给表加上读锁(共享锁)，自己可读，不可泄，其他人可读不可写，自己不可操作其他表
            lock tables table1 read
            -- 给表加上写锁（排他锁）,自己可读可写，其他人不可读不可写，自己不可操作其他表
            lock tables tables write;
            -- 释放锁
            unlock tables;
            ```

        2. 意向锁：由于innodb允许表锁与行锁共存，当表中某一行记录被加上行锁，那么数据库会在高一级的表锁中加上一个意向锁，表明其中有某条记录有锁，此时如果有其他事务想要加表锁就可以发现已经有锁了。如果一个事务对某行记录加上了写锁，那么表空间也会加上一个意向排他锁，如果加上的是读锁，那么表空间就会加上意向共享锁。

            ```sql
            -- 使用以下语句进行记录行的加锁操作时，会自动为这个表加上对应的意向锁，并让这个事务获取
            -- 这个会获取意向共享锁
            select ... LOCK IN SHARE MODE;
            select ... FOR SHARE 
            -- 这个会获取意向排他锁
            select ... FOR UPDATE
            ```

        3.  自增锁和元数据锁（Matedata lock）：自增锁就是在向表中插入数据的时候，如果插入的列有自增属性的，就会自动申请自增锁，这个是表锁。元数据锁，一般情况下，innodb表进行增删改查操作时会添加MDL读锁（自动），但是如果进行表结构修改的时候添加MDL写锁，修改表结构时会自动申请。

    - 行锁：Record lock，Gap lock，Next-Key lock，插入意向锁

        1. 记录锁（Record lock）：顾名思义就是在某条记录上加上锁，官方名称：LOCK_REC_NOT_GAP,记录锁也分为S锁和X锁

            ```sql
            -- 添加S锁记录所=锁
            select * from table1 where id=1 lock in share mode ;
            -- 添加X锁记录所
            select * from table1 where id=1 for update;
            update table1 set name='asd' where id =1;
            ```

        2. 间隙锁（Gap lock）：官方名称为：LOCAK_GAP，这个锁就是在某些空间中加上锁，因此不区别S/X,不同的事务可以重复添加间隙锁(相互之间会有影响，两个事务同时持有同一个间隙锁，如果同时进行插入操作会造成死锁，此时会选择一个成本较低的事务直接进行回滚，让另一个事务成功),只有在隔离级别为可重复读的情况下有有效，因为如果都不可重复读，那么肯定不支持间隙锁，间隙锁的存在就是为了解决不可重复读的问题。

            ```sql
            -- 如果id=3到id=8之间没有数据，那么以下的语句会添加(3，8）间隙锁,注意读写锁是互斥的
            select * from table1 where id =5 lock in share mode;
            -- 由于上方的sql添加了间隙锁，因此下面的sql执行会阻塞
            insert into table1(id,name) values(6,'lujie');
            ```

        3. 临键锁（Next-Key lock）：临键锁相当于记录锁与间隙锁的集合，

            ```sql
            -- 如果id=3到id=8之间没有数据，下面的锁就锁定了（3，8]
            select * from table1 where id <=8 and id>3 for update;
            -- 由于上方的sql添加了间隙锁，因此下面的sql执行会阻塞
            insert into table1(id,name) values(6,'lujie');
            ```

        4. 插入意向锁：官方名称：LOCK_INSERT_INTENTION

            ```sql
            -- 事务1，如果id=3到id=8之间没有数据，下面的锁就锁定了（3，8]
            select * from table1 where id <=8 and id>3 for update;
            -- 事务2运行以下操作并阻塞，会生成一个插入意向锁，表明要插入6这条数据
            insert into table1(id,name) values(6,'lujie');
            -- 事务3运行以下操作并阻塞，也会生成一个插入意向锁，表明要插入7这条数据，但是与上面的事务2的插入意向锁不冲突，可以并行
            insert into table1(id,name) values(7,'lujie');
            -- 当事务1结束，则事务2和3会同时成功
            ```

    - 页级锁：粒度介于表锁和行锁之间，效率也介于其之间，可能limit可以触发
    - 最后注意，每个层级的锁的数据是需要占用空间的，所以数量是有限制的，如果低层级的锁数量超出限制，则会升级为高级别的锁从而减少空间占用，但也会降低效率，例如频繁删除数据，可能会锁表。

- 按照锁的态度划分：悲观锁，乐观锁
    - 悲观锁是一种态度，就是不论做什么操作的时候都通过显式加锁
    - 乐观锁就是查询的时候不显式加锁
    - 注意通过*select ... for update*操作进行加锁的时候一定要确认使用到了索引，如果没有使用索引那么这个操作会进行全表扫描，所有扫描的记录都会被锁住
- 按照加锁方式：隐式锁，显式锁
    - 隐式锁：类似于插入意向锁，insert操作一般不需要锁，但是如果没有加锁，如果有别人来读取这条数据就会有问题，所以新增了插入意向锁就是隐式锁
- 其他：全局锁，死锁
    - 全局锁：*Flush table with read lock*
    - 死锁：两个事务操作中形成死锁，可以通过等待超时解决，也可以通过死锁检测进行解决（检测到死锁后会回滚成本最小的事务）
    - 如何减少死锁：
      1. 大事务拆成小事务
      2. 索引优化，尽量少扫描行
      3. 尽量避免修改操作在前面
      4. 尽量不要显式加锁
      5. 可以考虑降低隔离级别
  - 锁的结构：锁所在事务的信息（只是指针，指向真正的事务信息），索引信息（对于行锁需要记录这个锁是哪个索引的），表锁行锁信息（判断这个锁是那种类型），type_mode（锁的模式：S/X/IX/IS/自增锁，锁类型：表锁/行锁，锁的具体信息：如果是行锁才会有这个，间隙锁/记录锁/临键锁）,其他信息（is_waiting是否在等待）   

- 锁的监控

```sql
-- 查看行锁的情况
show status like 'innodb_row_lock%'
-- 查看所有正在运行的线程信息
show processlist
-- 查看正在等待锁的信息
select * from performance_schema.data_lock_waits;
-- 5.7版本事务和锁涉及的三张表：information_schema.INNODB_TRX/ information_schema.INNODB_LOCKS/information_schema.INNODB_LOCK_WAITS
select * from information_schema.INNODB_TRX;
-- 8.0版本的表为：information_schema.INNODB_TRX/performance_schema.INNODB_LOCKS/performance_schema.INNODB_LOCK_WAITS
select * from performance_schema.INNODB_LOCKS;
```

#### 多版本并发控制MVCC

- MVCC的实现依赖于：Read View ,Undo log ,隐藏字段

- 一致性读的实现主要通过两种方式：当前读和快照读
    - 当前读：就是加锁读，读取数据的时候进行加锁，读取到最新的数据
    - 快照读：简单的select就是快照读，读取的时快照数据，可以实现读写并发

- 隐藏字段：trx_id,roll_pointer
    - trx_id：表中数据的隐藏字段，记录着这条数据最近改动的事务
    - roll_pointer:表中的隐藏字段，指向前一个操作的undo日志
- Read View:creator_trx_id,trx_ids,up_limit_id,low_limit_id
    - creator_trx_id:创建这个Read View的事务id(表中的记录数据只有被修改的时候trx_id才会有值，查询的时候没有)
    - trx_ids:当前事务开启时还在活跃的事务列表（没有提交的事务）
    - up_limit_id:活跃事务中ID最小的
    - low_limit_id:全局系统中最大的事务ID+1
    - 主要流程：访问一条记录的时候判断这条记录的trx_id,如果等于creator_trx_id就可以访问，如果不在trx_ids列表中就可以访问，如果小于up_limit_id就可以访问，如果大于等于low_limit_id就不能访问，如果这条记录可以被访问就停止，如果不能访问就要根据roll_pointer寻找undo日志中的快照，直到找到或者没有。



### 日志

事务有四种特性，其中隔离性时通过加锁实现的，持久性通过Redo日志实现，一致性和原子性通过Undo日志实现

#### Redo日志

Redo日志记录的不是具体的sql操作，而是数据页中的数据该如何变化，具体指定了具体页数据的修改。事务在执行的过程中，不会每个事务提交之后都会及时刷到磁盘上，但是为了防止数据的丢失，就会在事务提交之后先将事务写到Redo日志中。如果之后程序奔溃还可以进行恢复，保证了持久性。

1. 首先Redo日志也分为两个部分，第一部分也是内存层面的缓冲区，可以通过以下的命令查看缓冲区的大小

```sql
show variables like 'log_buffer_size'
```

2. 其次是磁盘上的文件作为实体的redo日志文件：*ib_logfile0*和*ib_logfile1*
3. 事务在运行过程中，会在内存中更新数据，随后把这个操作同步到redo日志的缓冲区，然后redo日志的缓冲区通过操作系统的命令将缓冲区的数据写入到redo日志中，最后再将内存中的操作刷到磁盘中，注意redo日志只是作为可靠性的保证，实际如果程序没有问题，最终刷入磁盘的数据还是内存中的数据不是redo日志的
4. 根据3中的描述，可以看出最重要的就是redo缓冲区的数据写入redo日志中，只要这个步骤没有问题那么程序就不会丢失数据。因此，针对这个步骤最好是缓冲区只要有新增的操作就立即刷到redo日志。因此，mysql提供了*innodb_flush_log_at_trx_commit*参数进行配置
    - *0*：每次事务运行过程中，不论事务是否提交，redo缓存不会立即写入系统的page cache中，而page cache也不会刷到redo日志中，而是通过系统的后台线程每隔1s将redo缓存中数据刷到page cache然后立即再刷到redo日志中，所以可能会出现事务还没有提交，操作就已经到redo日志中了
    - *1*：默认值，事务运行过程中会不断刷数据到redo缓存中，一旦事务提交，就立马把缓存数据刷到page cache中然后再立即刷到redo日志中
    - *2*：每次事务运行过程中数据会不断刷到redo缓存中，当事务提交之后立马将缓存中的数据同步到page cache中，然后就不管了，交给系统啥时候刷到redo日志中

5. redo日志文件的写入逻辑类似于一个循环队列，通过两个指针指定位置，第一个指针指定数据从哪里开始写入，第二个指针指定从哪里开始之后的空间不能覆盖

#### Undo日志

Undo日志在整个事务的流程中处于redo日志之前，首先将数据库数据加载到内存中，然后将要更新的数据的旧值写入到undo日志中，然后更新内存数据，然后将内存数据写入redo缓存，再写入redo日志，最后将内存数据刷入磁盘中。

1. undo日志中不会保存select语句的相关信息
2. undo日志是逻辑回滚不是物理回滚，因为插入的数据不会物理删除，还是存在于磁盘中，只是再逻辑上进行了删除
3. MVCC是通过undo日志实现的
4. *innodb_undo_directory*:undo日志位置，