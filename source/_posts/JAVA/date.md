---
title: 日期类型
date: 2021-12-15 17:31:06
categories:
  - JAVA
tags:
  - Date
  - SimpleDateFormat
  - Calendar
author: Fanrencli
---
## Date
- 构造方法：new Date(long time) 将long型数据转为日期
- 普通方法：getTime(),返回一个long型日期
```java
    // 输出日期
    Date date= new Date();
    System.out.println(date);
    long cur = date.getTime();
```

## SimpleDateFormat类型用于日期的格式转换
- 构造函数：`new SimpleDateFormat("日期格式")`
- Date转String：`String Format(Date date)`
- String转Date：`Date prase(String str)`

```java
public class test {
    public static void main(String[] args) throws ParseException {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
        Date date = new Date();
        String str = simpleDateFormat.format(date);
        System.out.println(str);
        date = simpleDateFormat.parse(str);
        System.out.println(date);
    }
}
```

## Calendar日期类型主要用于计算
- 构造函数私有化
- 静态方法取得实例对象：getInstance()
- 静态属性：YEAR，MONTH，DAY_OF_MONTH，HOUR_OF_DAY，MINUTE，SECOND
```java
public class test {
    public static void main(String[] args) throws ParseException {
        Calendar calendar = Calendar.getInstance();
        System.out.println(calendar.get(Calendar.YEAR));
        System.out.println(calendar.get(Calendar.MONTH));
        System.out.println(calendar.get(Calendar.DAY_OF_MONTH));
        System.out.println(calendar.get(Calendar.HOUR_OF_DAY));
        System.out.println(calendar.get(Calendar.MINUTE));
        System.out.println(calendar.get(Calendar.SECOND));
    }
}
```