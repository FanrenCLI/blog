---
title: Collections
date: 2021-12-17 16:37:21
categories:
  - JAVA
tags:
  - Collections
author: Fanrencli
---

## 集合

- 类集就是动态对象数组
- `Collection`、`List`、`Set`
- `Map`
- `Iterator`、`Enumeration`

### Collections

- 集合的最大父接口：`public interface Collection<E> extends Iterator`
- 集合常用方法：
    - **添加一个元素：`public boolean add(E value)`;**
    - 追加一个集合：`public boolean addAll(Collection<? extends E> c)`;
    - 清空集合：`public void clear()`;
    - 判断是否包含元素：`public boolean contains(Object o)`;
    - 判断集合是否为空：`public boolean isEmpty()`;
    - 删除对象：`public boolean remove(Object o)`;
    - 取得集合大小：`public int size()`;
    - 将集合变为数组返回：`public Object[] toArray()`;
    - **为Iterator实例化：`public Iterator<E> iterator()`;**

### List（元素可以重复）

- `List`是`Collection`的子接口，在原有的基础上添加了许多方法，主要包含以下：
    - **取得索引对应的内容：`public E get(int index)`;**
    - 修改制定索引的内容：`public E set(int index, E value)`;
    - 为ListIterator进行实例化：`public ListIterator<E> listIterator()`;
- 常用实现子类：**`ArrayList`**,`Vector`

```java

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

class  Book{
    private double price;
    private String title;
    Book(String title,double price){
        this.price = price;
        this.title = title;
    }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Book book = (Book) o;
        return price == book.price &&
                Objects.equals(title, book.title);
    }

    @Override
    public int hashCode() {

        return Objects.hash(price, title);
    }

    @Override
    public String toString() {
        return "Book{" +
                "price=" + price +
                ", title='" + title + '\'' +
                '}';
    }

    public double getPrice() {
        return price;
    }

    public void setPrice(int price) {
        this.price = price;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }
}
public class Main {
    public static void main(String[] args)
    {
        List<Book> list = new ArrayList<>();
        list.add(new Book("java",220.2));
        list.add(new Book("git",225.231));
        list.add(new Book("SVN",12.22));
        list.remove(new Book("git",225.231));
        System.out.println(list);
        System.out.println(list.get(1));
    }
}
```

### Set（元素不可以重复）

- 集合判断元素是否重复需要类重写`equals`和`hashcode`方法
- `Set`接口没有在原有方法的基础上进行大量的扩充方法，只是简单的实现了集合接口
- 常用子类：`HashSet`(无序)、`TreeSet`（有序）
- `TreeSet`：进行排序时需要类实现`Comparable`接口，而`TreeSet`判断一个类是否重复就是通过`Comparable`接口的比较方法实现的，此处会出现异常；

### 集合输出方法

- `Iterator`
- `Enumeration`
- `ListIterator`
- `foreach`

```java
public class Main {
    public static void main(String[] args)
    {
        Set<String> set = new HashSet<>();
        set.add("HELLO");
        set.add("WORLD");
        set.add("HELLO");
        Iterator<?> it = set.iterator();
        while (it.hasNext()){
            System.out.println(it.next());
        }
    }
}
```

```java

public class Main {
    public static void main(String[] args)
    {
        List<String> set = new Vector<>();
        set.add("HELLO");
        set.add("WORLD");
        set.add("HELLO");
        Enumeration<String> enu = ((Vector<String>) set).elements();
        while (enu.hasMoreElements()){
            System.out.println(enu.nextElement());
        }
    }
}
```

```java
public class Main {
    public static void main(String[] args)
    {
        ArrayList<String> set = new ArrayList<>();
        set.add("HELLO");
        set.add("WORLD");
        set.add("HELLO");
        ListIterator<String> it = set.listIterator();
        while (it.hasNext()){
            System.out.println(it.next());
        }
        while (it.hasPrevious()){
            System.out.println(it.previous());
        }
    }
}
```

```java
public class Main {
    public static void main(String[] args)
    {
        ArrayList<String> set = new ArrayList<>();
        set.add("HELLO");
        set.add("WORLD");
        set.add("HELLO");
        for(String str: set){
            System.out.println(str);
        }
    }
}
```

### Map

-  `Map`接口定义了相关的函数方法:
    - 向集合中保存数据：`public V put(K key, V value)`;
    - 根据Key查找数据：`public V get(Object key)`;
    - 将Map集合转化为Set集合：`public Set<Map.Entry<K,V>> entrySet()`
    - 取出全部的Key：`public Set<K> keySet()`
- 常用子类：`HashMap`（key或value可以为空，线程不安全）,`HashTable`（key和value不能为空，线程安全）
- 无序存储，内容覆盖

```java
public class Main {
    public static void main(String[] args)
    {
        Map<String,Integer> map = new HashMap<>();
        map.put("yi",1);
        map.put("er",2);
        map.put("san",3);
        map.put("san",33);
        System.out.println(map);
    }
}
public class Main {
    public static void main(String[] args)
    {
        Map<String,Integer> map = new HashMap<>();
        map.put("yi",1);
        map.put("er",2);
        map.put("san",3);
        map.put(null,0);
        System.out.println(map.get("yi"));
        System.out.println(map.get(null));
    }
}
public class Main {
    public static void main(String[] args)
    {
        Map<String,Integer> map = new Hashtable<>();
        map.put("yi",1);
        map.put("er",2);
        map.put("san",3);
        System.out.println(map.get("yi"));
    }
}
public class Main {
    public static void main(String[] args)
    {
        Map<String,Integer> map = new Hashtable<>();
        map.put("yi",1);
        map.put("er",2);
        map.put("san",3);
        Set<Map.Entry<String,Integer>> set = map.entrySet();
        Iterator<Map.Entry<String, Integer>> it = set.iterator();
        while(it.hasNext()){
            Map.Entry<String, Integer> me=  it.next();
            System.out.println(me.getKey()+"="+me.getValue());
        }
    }
} 
```

### Stack

- `Stack`作为`Vector`的子类，只需要记得两个方法：入栈(put)和出栈(pop)

```java
public class Main {
    public static void main(String[] args)
    {
        Stack<String> list=  new Stack<>();
        list.push("asd");
        list.push("sss");
        System.out.println(list.pop());
    }
}
```

### Properties

- 设置属性：`public Object setProperty(String key,String Value)`;
- 取得属性：`public String getProperty(String key)`;
- 取得属性：`public String getProperty(String key, String defaultValue)`;
- 保存属性：`public void store(OutputStream out, String comments) `
- 读取属性：`public synchronized void load(InputStream in) throws IOException`

```java
public class Main {
    public static void main(String[] args) throws IOException {
        Properties pro = new Properties();
        pro.setProperty("a","1");
        pro.setProperty("b","2");
        pro.store(new FileOutputStream(new File("test.txt")),"this infor");
    }
}
public class Main {
    public static void main(String[] args) throws IOException {
        Properties pro = new Properties();
        pro.load(new FileInputStream(new File("test.txt")));
        System.out.println(pro);
    }
}
```

### Collections工具类

- 作为工具类，提供了一系列的集合操作方法

```java
public class Main {
    public static void main(String[] args) throws IOException {
        List<String> list= new ArrayList<>();
        Collections.addAll(list,"a","b","c");
        System.out.println(list);
        Collections.reverse(list);
        System.out.println(list);
    }
}
```






