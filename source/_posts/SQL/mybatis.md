---
title: Mybatis
date: 2022-10-25 14:11:00
categories:
  - JAVA
tags:
  - Mybatis
author: Fanrencli
---

### 批量查询
```xml
<select id="" parameterType="" resultType="">
    select id,name,code
    from Tablename
    where id in 
    <foreach collection="" item="item" index="index" separator="," open="(" close=")">
        #{item.id}
    </foreach>
</select>
```

### 删除数据
```xml
<delete databaseId='mysql' id=''>
    delete a.id from TBALENAME a,TABLENAME2 b
    where a.id=b.id
</delete>
```

### 连表新增
```xml
<insert databaseId='mysql' id = 'MethodName'>
    INSERT INTO TABLE1(
        id,
        name,
        age,
        address
    )
    SELECT
        table2.id,
        #{name}
        table2.age,
        #{address}
    from TABLE2 table2
    where
        table2.id='xx'
</insert>
```

### 多表更新
```xml
<update id=''>
    UPDATE TABLENAME tb1
    <set>
        tb1.id = 
        case
        when NOT EXISTS (SELECT 1 FROM TABLENAME2 tb2 WHERE tb1.id=tb2.id ) then 'X'
        when NOT EXISTS (SELECT 1 FROM TABLENAME2 tb2 WHERE tb1.id1=tb2.id1 ) then 'X'
        when NOT EXISTS (SELECT 1 FROM TABLENAME2 tb2 WHERE tb1.id2=tb2.id2 ) then 'X'
        end
    </set>
    WHERE tb.id='X'
</update>
```


### 批量更新
```xml
<update id="updateBatch"  parameterType="java.util.List">  
    <foreach collection="list" item="item" index="index" open="" close="" separator=";">
        update course
        <set>
            name=${item.name}
        </set>
        where id = ${item.id}
    </foreach>      
</update>
```

```xml
<update databaseId='oracle'>
    <foreach collection='rows' item='item' index='index' open='begin' close=';end;' separator=';'>
        merge into jc_torganization a
        using(select #{item.orgCode} as org_code from dual ) b
        on (a.org_code=b.org_code)
        when MATCHED THEN
        update set
        enterprise_nature=nvl(#{item.enterpriseNature}, a.enterprise_nature),
        owner_industry=nvl(#{item.ownerIndustry}, a.owner_industry)
        when NOT MATCHED THEN
        xxx
    </foreach>
</update>
```

```xml
<update id="TableName" parameterType="list">
    update TableName
    <trim prefix="set" suffixOverrides=",">
        <trim prefix="FIELD_NAME_1 =case" suffix="end,">
            <foreach collection="list" item="item" index="index">
                <if test="item.FIELDNAME1!=null">
                    when XX_id=#{item.XXId}
                        then #{item.FIELDNAME1}
                </if>
            </foreach>
        </trim>
        <trim prefix="FIELD_NAME_2 =case" suffix="end,">
            <foreach collection="list" item="item" index="index">
                <if test="item.FIELDNAME2!=null">
                    when XX_id=#{item.XXId}
                    then #{item.FIELDNAME2}
                </if>
            </foreach>
        </trim>
    </trim>
    where XXid in
    <foreach collection="list" item="item" index="index" separator="," open="(" close=")">
        #{item.XXid}
    </foreach>
</update>
```
### 问题

#### GROUP BY
- select语句查询的字段如果没有使用聚合函数，必须出现在group by后，否则报错
```xml
<select>
select a.id,a.name,a.code,max(a.count),max(a.date) from TABLE a
where a.id=xxx
group a.id,a.name,a.code
</select>
```

#### NULL或空
- 更新数据时，针对字符串数据需要判断为空或NULL（oracle将''或NULL都认为是NULL，mysql允许为''）
```xml
<update datebaseid='oracle'>
update TABLENAME a
SET
a.id = nvl(#{id},a.id)
where a.id=xxx
</update>
<update datebaseid='mysql'>
update TABLENAME a
SET
<choose>
    <when test='id!=null and id !='''>
        id=#{id}
    </when>
    <otherwise>
        id=a.id
    </otherwise>
</choose>
where a.id=xxx
</update>
```

#### where 不能使用列假名,where语句先于假名出现执行
```xml
<select>
select id, 10 count from TABLENAME
where count>1
</select>
```

#### where不能使用聚合函数
```xml
<select>
select id, 10 count from TABLENAME
where sum(id) =10
</select>
```
#### having用于集合函数的过滤
```xml
<select>
select id, 10 count from TABLENAME
having sum(id) =10
</select>
```
#### row_number() over(Partition by xxx order by xxx)使用方法
```xml
<select>
select id,name,code, row_number() over(partition by id,name order by code) rn
from tablename
</select>
```
| id | name | code | rn |
| :-----| :----- | :----- | :----- |
| 1 | a | 1 |1 |
| 1 | a | 2 |2 |
| 1 | a | 5 |3 |
| 2 | b | 2 |1 |
| 2 | b | 12 |2 |
| 3 | c | 2 |1 |
| 3 | c | 12 |2 |
| 3 | c | 232 |3 |
| 3 | c | 1123 |4 |
| 4 | d | 2 |1 |

#### SQL执行顺序,解释了为何where不能够使用别名

```xml
    From Table1,Table2
    join Table3 on ...
    where ...
    group by ...
    having...
    select xxx
    order by ...
    limit ...
```

#### 存储过程
```xml
<!-- 示例 -->
    delimiter $ 定义结束符号
    create procedure select_all_data(in name1 varchar(20),out sex1 varchar(10))
    begin
        select sex into sex1 from users where name = name1;
    end $
    delimiter ; 防止冲突，重新定义
<!-- 上面定义了一个存储过程 -->
    set @name1= "lj"
    call select_all_data(@name1,@sex1);
    select @sex1 from dual;
```

#### 外键约束

```xml
CREATE TABLE IF NOT EXISTS test1(
    id int primary key,
    `name` varchar(10) not null DEFAULT('') UNIQUE,
    sex TINYINT not null DEFAULT('1')
);
CREATE table test2(
    id int PRIMARY key,
    relation_id int not null UNIQUE,
    CONSTRAINT uk_f_id FOREIGN key (relation_id) REFERENCES test1 (id)
)
```