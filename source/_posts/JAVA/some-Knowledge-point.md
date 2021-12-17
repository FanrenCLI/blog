---
title: some_Knowledge_point
date: 2021-12-17 11:33:17
categories:
  - JAVA
tags:
  - Serializable
author: Fanrencli
---

### 序列化（Serializable）
- `Serializable`作为序列化的标识接口没有实现任何方法
- `transient`关键字作为防止序列化的关键字，能够使数据不被序列化
- `ObjectOutputStream`和 `ObjectInputStream`简易的实现了对象的序列化和反序列化操作
```java
class  B implements Serializable{
    private transient int a=1;
    private String title = "asd";
}
public class Main {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File("test.txt")));
        oos.writeObject(new B());
        oos.close();
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File("test.txt")));
        Object obj =  ois.readObject();
        ois.close();
    }
}
```

### 克隆（Cloneable）
