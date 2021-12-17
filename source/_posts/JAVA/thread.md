---
title: thread
date: 2021-12-15 22:02:10
categories:
  - JAVA
tags:
  - Thread
author: Fanrencli
---
## Thread线程

### 线程创建方式

- 继承`Thread`类
- 实现`Runnable`接口
- 实现`Callable`接口,`Callable`接口可以通过实现`call()`方法实现方法运行，并通过`FutureTask`来获取返回结果
  
```java
// 通过继承Thread类实现Run方法，并调用start方法运行。通过start方法运行可以交给CPU进行资源分配。
class A extends Thread{
    @Override
    public void run() {
        System.out.println("ok");
    }
}
public class test {
    public static void main(String[] args) {
        A a = new A();
        a.start();
    }
}
```
```java
// 通过实现Runnable接口，重写run方法，然后实例化对象借助Thread实力进行运行。
class A implements Runnable{
    @Override
    public void run() {
        System.out.println("ok");
    }
}
public class test {
    public static void main(String[] args) {
        new Thread(new A()).start();
    }
}
```
```java
class A implements Callable<String>{
    @Override
    public String call() throws Exception {
        return "ok";
    }
}
public class test {
    public static void main(String[] args) {
        FutureTask<String> futureTask = new FutureTask<>(new A());
        new Thread(futureTask);
    }
}
```

### 线程的相关操作

- `Thread.sleep()` 使得当前线程休眠一定的时间，是`Thread`的静态函数，不论谁调用`sleep`方法，休眠的总是当前线程。
- `getPriority()`和`setPriority(int newPriority)`,获取线程的优先级
- `join()`，通过线程实例对象调用join方法，使得当前线程等待join线程结束
- `yield()`方法使得当前线程让出CPU资源；
- `interrupt()`通过调用此方法发出一个信号，通常用于在线程阻塞时通知退出阻塞；
- `Object.wait()` 作为`Object`对象的方法，用于休眠当前线程，通过线程的对象调用，并放弃当前持有的锁，必须在`synchronized`中使用。
- `Object.notify[all]()` 作为`Object`对象的方法，用于唤醒等待此对象的线程，必须在`synchronized`中使用。


### Synchronized关键字作用

1. 修饰一个代码块，被修饰的代码块称为同步语句块，作用的对象是调用这个代码块的对象。
2. 修饰一个方法，被修饰的方法称为同步方法，作用的对象是调用这个方法的对象。
3. 修饰一个静态的方法，其作用的范围是整个静态方法，作用的对象是这个类的所有对象。
4. 修饰一个类，其作用的范围是synchronized后面括号括起来的部分，作用的对象是这个类的所有对象。
5. 总结：被`synchronized`修饰的对象，其所有synchronized方法被锁住，非synchronized方法正常使用。


```java
// synchronized修饰代码块，只锁定指定的对象的synchronized修饰的代码。this指当前对象，当有线程进入this对应的代码块，则此对象的所有synchronized修饰的代码全部锁住，非synchronized修饰的代码块任然可以进入。
public class A implements Runnable {
    public static void main(String[] args) throws InterruptedException {
    }
    Object lock1 = new Object();
    Object lock2 = new Object();
    @Override
    public void run() {
        // 第一把锁
        synchronized (lock1) {
            System.out.println("我是lock1，我叫"+ Thread.currentThread().getName());
            try {
                Thread.sleep(3000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println(Thread.currentThread().getName()+"lock1运行结束");
        }
        // 第二把锁
        synchronized (lock2) {
            System.out.println("我是lock2，我叫"+ Thread.currentThread().getName());
            try {
                Thread.sleep(3000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println(Thread.currentThread().getName()+"lock2运行结束");
        }
        synchronized (this) {
            System.out.println("我是this，我叫"+ Thread.currentThread().getName());
            try {
                Thread.sleep(3000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println(Thread.currentThread().getName()+"this运行结束");
        }
    }
}
```
```java
// 当synchronized修饰方法的时候，如果有县城进行这个某个实例对象的方法中，那么这个实例对象其他被synchronized修饰的方法同时也会被锁住。
class A{
    public synchronized void say(B b){
        System.out.println("sayA");
        b.get();
    }
    public synchronized void get(){
        System.out.println("getA");
    }
}
class B{
    public synchronized void say(A a){
        System.out.println("sayB");
        a.get();
    }
    public synchronized void get(){
        System.out.println("getB");
    }
}
public class Main {
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        A a = new A();
        B b = new B();
        new Thread(()->a.say(b)).start();
        b.say(a);
    }
```