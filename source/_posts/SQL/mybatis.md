---
title: Mybatis
date: 2023-12-16 14:11:00
categories:
  - JAVA
tags:
  - Mybatis
author: Fanrencli
---
## Mybatis高级


## Mybatis使用

### 批量查询
```sql
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
```sql
<delete databaseId='mysql' id=''>
    delete a.id from TBALENAME a,TABLENAME2 b
    where a.id=b.id
</delete>
```

### 连表新增
```sql
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
```sql
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
```sql
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

```sql
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

```sql
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

#### Oracle列名和表明可以使用双引号，但是字符串需要使用单引号

#### GROUP BY
- select语句查询的字段如果没有使用聚合函数，必须出现在group by后，否则报错
```sql
<select>
select a.id,a.name,a.code,max(a.count),max(a.date) from TABLE a
where a.id=xxx
group a.id,a.name,a.code
</select>
```

#### NULL或空
- 更新数据时，针对字符串数据需要判断为空或NULL（oracle将''或NULL都认为是NULL，mysql允许为''）
```sql
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
```sql
<select>
select id, 10 count from TABLENAME
where count>1
</select>
```

#### where不能使用聚合函数
```sql
<select>
select id, 10 count from TABLENAME
where sum(id) =10
</select>
```
#### having用于集合函数的过滤
```sql
<select>
select id, 10 count from TABLENAME
having sum(id) =10
</select>
```
#### row_number() over(Partition by xxx order by xxx)使用方法
```sql
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

#### SQL执行顺序,解释了为何where不能够使用`列的`别名

```sql
    From Table1,Table2
    join Table3 on ...
    where ...
    group by ...
    having...
    select xxx
    order by ...
    limit ...
```

#### Oracle批量插入
```sql
insert all into Student(id,name,sex)
into Student(id,name,sex) values ('004','zs','男')
into Student(id,name,sex) values ('005','lk','男')
select '006','ws','女' from dual;
```

### 窗口函数（8.0版本）

- ROW_NUMBER() over( partition by id order by name)

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

- RANK() over( partition by id order by name)

| id | name | code | rn |
| :-----| :----- | :----- | :----- |
| 1 | a | 1 |1 |
| 1 | a | 2 |1 |
| 1 | a | 5 |3 |
| 2 | b | 2 |1 |
| 2 | b | 12 |1 |
| 3 | c | 2 |1 |
| 3 | c | 12 |1 |
| 3 | c | 232 |1 |
| 3 | d | 1123 |4 |
| 4 | d | 2 |1 |

- DENSE_RANK() over( partition by id order by name)

| id | name | code | rn |
| :-----| :----- | :----- | :----- |
| 1 | a | 1 |1 |
| 1 | a | 2 |1 |
| 1 | a | 5 |2 |
| 2 | b | 2 |1 |
| 2 | b | 12 |1 |
| 3 | c | 2 |1 |
| 3 | c | 12 |1 |
| 3 | c | 232 |1 |
| 3 | d | 1123 |2 |
| 4 | d | 2 |1 |