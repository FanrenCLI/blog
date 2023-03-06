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
        new Thread(futureTask).start();
    }
}
```

### 线程池相关操作

- Future和FutureTask
- 线程池的创建
- 线程拒绝策略

#### Future 和 FutureTask

- Future常常作为线程池中submit方法的返回值的接收，可以通过调用get()方法获取线程返回值；
- FutureTask类实现了Runnable和Future接口，其构造函数可以接收Runnable和Callable实现类，然后可以通过线程池的submit或者execute方法进行运行,也可以直接通过Thread运行，最后可以调用get方法获取返回值；


```java
class A implements Callable<String> {
    @Override
    public String call() throws Exception {
        return "ok";
    }
}
class B implements Runnable{
    @Override
    public void run() {
        System.out.println("ok");
    }
}
public class Main {
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        FutureTask<String> futureTask = new FutureTask<>(new A());
        new Thread(futureTask).start();
        FutureTask<String> futureTask1 = new FutureTask<>(new B(),"backinfo");
        new Thread(futureTask1).start();
        System.out.println(futureTask1.get());
        System.out.println(futureTask.get());
    }
}
```

```java
ExecutorService service = Executors.newSingleThreadExecutor();
Future<String> future = service.submit(new Callable<String>() {
    @Override
    public String call() throws Exception {
        return "say helloWorld!!!";
    }
});
System.out.println(future.get());// 通过get返回结果

```
#### 线程池的创建

- `newCachedThreadPool`：重用以前线程，无可用则创建，空闲则移除；
- `newFixedThreadPool(int size)`：创建指定大小的线程池。通过队列管理
- `newScheduledThreadPool(int size)`:可以设置运行几次以及时间间隔
- `newSingleThreadExecutor`：创建只有一个线程的线程池，这个线程池可以在线程死后（或发生异常时）重新启动一个线程来替代原来的线程继续执行下去
- 线程池的运行方法有两种：
    - submit：返回一个future类型，可以获取返回值
    - execute：直接运行，无返回值

```java
ExecutorService pool = Executors.newFixedThreadPool(taskSize);
List<Future> list = new ArrayList<Future>(); 
for (int i = 0; i < taskSize; i++) { 
    Callable c = new MyCallable(i + " "); 
    Future f = pool.submit(c); 
    list.add(f); 
} 
for (int i = 0; i < taskSize; i++) { 
    Runnable c = new Myrunnable(i + " "); 
    Future f = pool.submit(c,:"backInfo"); 
    list.add(f); 
} 
pool.shutdown(); 
for (Future f : list) { 
    System.out.println("res：" + f.get().toString()); 
    }
```

#### 线程拒绝策略

<p style="text-indent:2em">
线程池中的线程已经用完了，无法继续为新任务服务，同时，等待队列也已经排满了，再也塞不下新任务了。这时候我们就需要拒绝策略机制合理的处理这个问题。
</p>

- `AbortPolicy`：直接抛出异常，阻止系统正常运行。 
- `CallerRunsPolicy`：只要线程池未关闭，该策略直接在调用者线程中，运行当前被丢弃的任务。显然这样做不会真的丢弃任务，但是，任务提交线程的性能极有可能会急剧下降。 
- `DiscardOldestPolicy`：丢弃最老的一个请求，也就是即将被执行的一个任务，并尝试再次提交当前任务。 
- `DiscardPolicy`：该策略默默地丢弃无法处理的任务，不予任何处理。如果允许任务丢失，这是最好的一种方案。 
- 以上内置拒绝策略均实现了RejectedExecutionHandler接口，若以上策略仍无法满足实际需要，完全可以自己扩展RejectedExecutionHandler接口。

### 线程的相关操作

- `Thread.sleep()` 使得当前线程休眠一定的时间，放弃CPU使用权，但是不会放弃资源锁，是`Thread`的静态函数，不论谁调用`sleep`方法，休眠的总是当前线程。
- `getPriority()`和`setPriority(int newPriority)`,获取线程的优先级
- `join()`，通过线程实例对象调用join方法，使得当前线程等待join线程结束
- `Thread.yield()`方法使得当前线程让出CPU资源；
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

### ThreadLocal详解

由于一个java项目中经常需要多线程操作且线程的创建与销毁都会产生CPU开销，所以在代码开发过程中，常用线程池进行多线程的操作。因此存在大量的线程复用的情况，一个线程的生命周期较长。在Thread线程的类中
```java
class Thread{
    ThreadLocal.ThreadLocalMap map = null;
}
class ThreadLocal{
    // 静态内部类
    static class ThreadLocalMap{
        // 静态内部类采用弱引用
        static class Entry extends WeakReference<ThreadLocal<?>>{

        }
    }
}
class Demo{
    // threadlocal使用时首先初始化一个ThreadLocal对象，然后通过set方法进行数据存储，从而达到线程之间隔离的效果
    // 此时，在这个线程中就会存在一个ThreadLocalMap<ThreadLocal,Integer>,其中key为创建的threadlocal对象，value为设置的值，
    // 使用完后，如果没有remove，就会导致内存泄漏的问题，
    ThreadLocal<Integer> threadlocal = ThreadLocal.withinit(()=>{ return 0;});
    threadlocal.set(1);
    threadlocal.get();
    threadlocal.remove();
}
```