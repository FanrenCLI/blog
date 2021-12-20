---
title: some_Knowledge_point
date: 2021-12-17 11:33:17
categories:
  - JAVA
tags:
  - Serializable
  - Cloneable
  - Comparable
  - Comparator
author: Fanrencli
---

### Lambda

- 实现一个接口的方法函数，接口只能有一个函数；

```java
interface IMessage{
    public int add(int x, int y);
}

public class Demo{
    public static void main(String[] args){
        IMessage msg = (s,y) -> a+b;
    }
}
```

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

- `Cloneable`作为标识接口不做任何方法的实现，表示类可以被克隆

```java

class Book implements Cloneable{
    private String title;
    private double price;
    public Book(String title, double price){
        this.title = title;
        this.price = price;
    }
    public getTitle() { return title; }
    public setTitle(String title) {this.title  = title; }
    public getPrice() { return price; }
    public setPrice(double price) {this.price = price; }

    @Override
    public object clone() throws ClassNotFoundException{ 
        return super.clone();
    }
}

public class Main {
    public static void main(String[] args) {
        Book book1 = new Book("java",22.2);
        Book book2 = (Book) book1.clone();
    }
}
```

### 比较器（Comparable/Comparator）

- `Comparable`是一个接口类，通过在定义类的时候实现接口函数来实现对象的比较，常用于集合中排序
- `Comparator`是一个比较的工具接口，用于对已经实现完成的类进行比较

```java
public interface Comparable<T>{
    public itn compareTo(T o);
}

class Book implements Comparable<Book> {
    private String title;
    private double price;
    public Book(String title, double price){
        this.title = title;
        this.price = price;
    }
    public getTitle() { return title; }
    public setTitle(String title) {this.title  = title; }
    public getPrice() { return price; }
    public setPrice(double price) {this.price = price; }

    @Override
    public int compare(Book value1, Book value2){
        if(value.getPrice() > value2.getPrice()){
            return 1;
        }else if(value1.getPrice() < value2.getPrice()){
            return -1;
        }esle{
            return 0;
        }
    }
}

public class Demo{
    public static void main(String args[]){
        Book books[] = new Book[]{ new Book("java",99.5),new Book("git",99.4),new Book("SVN",55.6)};
        Arrays.sort(books)
    }
}

```

```java

@FunctionalInterface
public interface Comparator<T>{
    public int compare(T value1, T value);
    public boolean equals(Object obj);
}

class Book{
    private String title;
    private double price;
    public Book(String title, double price){
        this.title = title;
        this.price = price;
    }
    public getTitle() { return title; }
    public setTitle(String title) {this.title  = title; }
    public getPrice() { return price; }
    public setPrice(double price) {this.price = price; }
}
class BookComparator implements Comparator<Book>{
    @Override
    public int compare(Book value1, Book value2){
        if(value.getPrice() > value2.getPrice()){
            return 1;
        }else if(value1.getPrice() < value2.getPrice()){
            return -1;
        }esle{
            return 0;
        }
    }
}
public class Demo{
    public static void main(String args[]){
        Book books[] = new Book[]{ new Book("java",99.5),new Book("git",99.4),new Book("SVN",55.6)};
        Arrays.sort(books,new BookComparator())
    }
}
```

### 方法引用

- 引用静态方法：类名::static方法名称
- 引用某个对象的方法：实例化对象::普通方法
- 引用特定类型的方法：特定类:: 普通方法
- 引用构造方法：类名称:: new
- 方法引用的接收接口只能实现一个方法，这就引出了`@FunctionalInterface`所定义的四种接口
    - 功能型接口：`public interface Fcuntion<T,R> {public R apply(T t)}`
    - 消费型接口：`public interface Consumer<T> {public void accept(T t)}`
    - 供给型接口：`public interface Supplier<T> {public T get()}`
    - 断言型接口: `public interface Predicate<T> {public boolean apply(T t)}`

```java
@FunctionalInterface
interface IMessage<P,R>{
    public R zhuanhuan(P p);
}

public class Main {
    public static void main(String[] args) {
        IMessage<Integer,String> msg = String::valueOf;
        System.out.println(msg.zhuanhuan(1000));
    }
}

```

```java
@FunctionalInterface
interface IMessage<R>{
    public R upper();
}

public class Main {
    public static void main(String[] args) {
        IMessage<String> msg = "hello"::toUpperCase;
        System.out.println(msg.upper());
    }
}

```

```java
@FunctionalInterface
interface IMessage<P>{
    public int compare(P p1, P p2);
}

public class Main {
    public static void main(String[] args) {
        IMessage<String> msg = String:: compareTo;
        System.out.println(msg.compare("A", "B"));
    }
}

```

```java

@FunctionalInterface
interface IMessage<P>{
    public P create(String title,double price);
}

class Book{
    private String title;
    private double price;
    public Book(String title, double price){
        this.title = title;
        this.price = price;
    }
    public getTitle() { return title; }
    public setTitle(String title) {this.title  = title; }
    public getPrice() { return price; }
    public setPrice(double price) {this.price = price; }
}

public class Main {
    public static void main(String[] args) {
        IMessage<Book> msg = Book :: new;
        System.out.println(msg.create("java", 20.2));
    }
}

```









