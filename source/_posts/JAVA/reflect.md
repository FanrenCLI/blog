---
title: Reflect
date: 2021-12-15 20:23:48
categories:
  - JAVA
tags:
  - reflect
author: Fanrencli
---
## 反射

### 反射3种实例化方法

1. 使用`实例化对象.getclass()`方法；
2. `类.class`
3. 使用`Class.forName(String str)`方法

### 通过反射实例化对象

通过`newinstance()`方法获取类的实例化对象相比于new方法虽然麻烦，但是可以不在需要手工去创建对象。在项目维护后期，如果需要添加一定量的类，可以通过制定类的全路径名来反射创建类的实例化对象，而不再需要去更改工厂类的代码。

```java
class B{
    public B(){
        System.out.println("ok");
    }

    @Override
    public String toString() {
        return "ok1";
    }
}
public class test1 {
    public static void main(String[] args) throws Exception {
        Class <?> cls = Class.forName("package.B");
        Object obj = cls.newInstance();
        B b = (B) obj;
        System.out.println(b);
    }
}
```
```java
class C{
    C(){
        System.out.println("C");
    }
}
class Factory{
    public static Object getInstance(String str) throws Exception  {

        return Class.forName(str).newInstance();
    }
}
public class test1 {
    public static void main(String[] args) throws Exception {
        Object obj = Factory.getInstance("demo3.C");
    }
}
```
### 通过反射调用构造函数

- `getConstructor(参数类...)` 获取指定的构造函数

```java
class C{
    private int a=0;
    C(int a ){
        this.a = a;
    }
}
public class test1 {
    public static void main(String[] args) throws Exception {
        Class<?> cls = Class.forName("demo3.C");
        Constructor<?> con = cls.getConstructor(int.class);
        Object obj = con.newInstance(12);
    }
}
```
### 通过反射调用普通函数

- `getMethod()`获取指定方法（不包含私有方法）
- `getDeclaredMethod()`获取指定方法
- `getMethods()`获取所有方法（不包含私有方法）
- `getDeclaredMethods()`获取所有方法

```java
class C{
    public int get(){
        System.out.println("OK");
        return 1;
    }
    public void set(String str){
        System.out.println(str);
    }
}
public class test1 {
    public static void main(String[] args) throws Exception {
        Class<?> cls = Class.forName("demo3.C");
        Object obj = cls.newInstance();
        Method setMe = cls.getMethod("set",String.class);
        Method getMe = cls.getMethod("get");
        setMe.invoke(obj,"set");
        getMe.invoke(obj);
    }
}
```

### 通过反射获取成员变量

- `getField(String str)`获取指定成员（不包含私有）
- `getDeclaredField(String str)`获取指定成员
- `getFields(String str)`获取所有成员（不包含私有）
- `getDeclaredFields(String str)`获取所有成员

```java
class C{
    private int a=1;

}
public class test1 {
    public static void main(String[] args) throws Exception {
        Class<?> cls = Class.forName("demo3.C");
        Object obj = cls.newInstance();
        Field field = cls.getDeclaredField("a");
        field.setAccessible(true);
        field.set(obj,12);
    }
}
```
