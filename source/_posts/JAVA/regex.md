---
title: Regex
date: 2021-12-15 17:31:06
categories:
  - JAVA
tags:
  - Regex
author: Fanrencli
---

## 正则标记

### 单个字符

-  `\\`:表示转义字符“\”;
-  `\t`:表示一个“\t”符号;
-  `\n`:匹配换行“\n”符号;


### 字符集

- `[abc]`:表示可能是字符a或者是字符b或者是字符c中的任意一个；
- `[^abc]`:表示不是a、b、c中的任意一位；
- `[a-z]`:表示任一的小写字母；
- `[a-zA-Z]`:表示任意一位字母，不区分大小写；
- `[0-9]`:表示任意的一位数字；


### 简化的字符集表达式

- `.`:表示任意的一位字符；
- `\d`:等价于`[0-9]`;
- `\D`:等价于`[^0-9]`;
- `\s`:表示任意的空白字符，如：`\t`,`\n`;
- `\S`:表示任意的非空白字符；
- `\w`:等价于`[a-zA-Z_0-9]`,表示由任意的字母，数字，_，所组成；
- `\W`:等价于`[^a-zA-Z_0-9]`,表示不是有任意的字母，数字，_所组成；


### 边界匹配（通常用于Javascript）

- `^`:正则的开始；
- `$`:正则的结束；


### 数量表达

- `?`:表示此正则可以出现0次或者1次；
- `+`:表示此正则可以出现1次或者1次以上；
- `*`:表示此正则可以出现0次/1次或多次；
- `{n}`:表示此正则正好出现n次；
- `{n,}`:表示次正则出现n次以上（包含n次）；
- `{n,m}`:表示此正则出现n~m次；


### 逻辑运算

- 正则1正则2：正则1判断后继续判断正则2；
- 正则1|正则2：正则1或正则2满足一个就可以；
- （正则）：将正则视为一组，同时可以先设置出现次数；

## 代码示例

```java
// 取出所有小写字母
public class test1 {
    public static void main(String[] args) {
        String str = "asdA)(S)(ASDsda)(&&*^%*%sdas1231532";
        String regex = "[^a-z]";
        System.out.println(str.replaceAll(regex,""));
    }
}

```
```java
// 按照数字拆分字符串
public class test1 {
    public static void main(String[] args) {
        String str = "jhgvfkjhg1341jhgv4jlhv1234vjhv1j234vlkgv13";
        String regex = "\\d+";
        String result[] = str.split(regex);
        for(String a:result){
            System.out.println(a);
        }
    }
}

```
```java
// 判断字符串是否是double型
public class test1 {
    public static void main(String[] args) {
        String str = "10.22";
        String regex = "\\d+(\\.\\d*)?";
        System.out.println(str.matches(regex));
    }
}

```
```java
// 判断是否是IP地址
public class test1 {
    public static void main(String[] args) {
        String str = "192.168.5.1";
        String regex = "(\\d{1,3}\\.){3}\\d{1,3}";
        System.out.println(str.matches(regex));
    }
}
```
```java
// 判断是否是日期
public class test1 {
    public static void main(String[] args) throws Exception {
        String str = "2009-03-12";
        String regex = "\\d{4}-\\d{2}-\\d{2}";
        System.out.println(str.matches(regex));
        if(str.matches(reges)){
          Date date = new SimpleDateFormat("yyyy-MM-dd").parse(str);
          System.out.println(date);
        }
    }
}
```
```java
// 判断电话号码：12345678|010-12345678|(010)-12345678
public class test1 {
    public static void main(String[] args) throws Exception {
        String str = "12345678";
        String regex = "\\d{7,8}|\\d{3,4}-\\d{7,8}|\\(\\d{3,4}\\)-\\d{7,8}";
        System.out.println(str.matches(regex));
    }
}
```
```java
// 判断邮箱
public class test1 {
    public static void main(String[] args) throws Exception {
        String str = "fanrencli@163.com";
        String regex = "[a-zA-Z]\\w{0,28}[a-zA-Z0-9]@\\w+\\.(net|cn|com\\.cn|com)";
        System.out.println(str.matches(regex));
    }
}

```