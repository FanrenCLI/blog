---
title: Lock
date: 2021-12-20 16:50:00
categories:
  - JAVA
tags:
  - Lock
author: Fanrencli
---

## 锁相关的概念

- 乐观锁
    - 认为读多写少，遇到并发写的可能性低，每次去读取数据的时候不会上锁，但是在更新的时候会判断一下在此期间别人有没有去更新这个数据，采取在写时先读出当前版本号，然后加锁操作（比较跟上一次的版本号，如果一样则更新），如果失败则要重复读-比较-写的操作。通过CAS操作实现的，CAS是一种更新的原子操作，比较当前值跟传入值是否一样，一样则更新，否则失败。
- 悲观锁
    - 认为写多，遇到并发写的可能性高，每次在读写数据的时候都会上锁，这样别人想读写这个数据就会block直到拿到锁。java中的悲观锁就是Synchronized。
- 自旋锁
    - 如果持有锁的线程能在很短时间内释放锁资源，那么那些等待竞争锁的线程就不需要做内核态和用户态之间的切换进入阻塞挂起状态，它们只需要等一等（自旋），等持有锁的线程释放锁后即可立即获取锁，这样就避免用户线程和内核的切换的消耗


## Synchronized同步锁

- Wait Set：调用wait方法被阻塞的线程被放置在这里； 
- Contention List：竞争队列，所有请求锁的线程首先被放在这个竞争队列中； 
- Entry List：Contention List中那些有资格成为候选资源的线程被移动到Entry List中； 
- OnDeck：任意时刻，最多只有一个线程正在竞争锁资源，该线程被成为OnDeck； 
- Owner：当前已经获取到所资源的线程被称为Owner； 
- !Owner：当前释放锁的线程。

![Synchronized实现原理](http://39.106.34.39:4567/20211222223047.png)

- Owner线程会在unlock时，将ContentionList中的部分线程迁移到EntryList中，并指定EntryList中的某个线程为OnDeck线程（一般是最先进去的那个线程）。
- Owner线程并不直接把锁传递给OnDeck线程，而是把锁竞争的权利交给OnDeck，OnDeck需要重新竞争锁。这样虽然牺牲了一些公平性，但是能极大的提升系统的吞吐量，在JVM中，也把这种选择行为称之为“竞争切换”。
- OnDeck线程获取到锁资源后会变为Owner线程，而没有得到锁资源的仍然停留在EntryList中。如果Owner线程被wait方法阻塞，则转移到WaitSet队列中，直到某个时刻通过notify或者notifyAll唤醒，会重新进去EntryList中。
- 处于ContentionList、EntryList、WaitSet中的线程都处于阻塞状态，该阻塞是由操作系统来完成的（Linux内核下采用pthread_mutex_lock内核函数实现的）。
- Synchronized是非公平锁。 Synchronized在线程进入ContentionList时，等待的线程会先尝试自旋获取锁，如果获取不到就进入ContentionList，这明显对于已经进入队列的线程是不公平的，还有一个不公平的事情就是自旋获取锁的线程还可能直接抢占OnDeck线程的锁资源。
- 每个对象都有个monitor对象，加锁就是在竞争monitor对象，代码块加锁是在前后分别加上monitorenter和monitorexit指令来实现的，方法加锁是通过一个标记位来判断的


## Lock接口

- `void lock()`: 执行此方法时, 如果锁处于空闲状态, 当前线程将获取到锁. 相反, 如果锁已经被其他线程持有, 将禁用当前线程, 直到当前线程获取到锁. 
- `boolean tryLock()`：如果锁可用, 则获取锁, 并立即返回true, 否则返回false. 该方法和lock()的区别在于, tryLock()只是"试图"获取锁, 如果锁不可用, 不会导致当前线程被禁用, 当前线程仍然继续往下执行代码. 而lock()方法则是一定要获取到锁, 如果锁不可用, 就一直等待, 在未获得锁之前,当前线程并不继续向下执行. 
- `void unlock()`：执行此方法时, 当前线程将释放持有的锁. 锁只能由持有者释放, 如果线程并不持有锁, 却执行该方法, 可能导致异常的发生. 
- `Condition newCondition()`：条件对象，获取等待通知组件。该组件和当前的锁绑定，当前线程只有获取了锁，才能调用该组件的await()方法，而调用后，当前线程将缩放锁。 
- `getHoldCount()` ：查询当前线程保持此锁的次数，也就是执行此线程执行lock方法的次数。 
- `getQueueLength()`：返回正等待获取此锁的线程估计数，比如启动10个线程，1个线程获得锁，此时返回的是9 
- `getWaitQueueLength(Condition condition)`：返回等待与此锁相关的给定条件的线程估计数。比如10个线程，用同一个condition对象，并且此时这10个线程都执行了condition对象的await方法，那么此时执行此方法返回10 
- `hasWaiters(Condition condition)`：查询是否有线程等待与此锁有关的给定条件(condition)，对于指定contidion对象，有多少线程执行了condition.await方法 
- `hasQueuedThread(Thread thread)`：查询给定线程是否等待获取此锁 
- `hasQueuedThreads()`：是否有线程等待此锁 
- `isFair()`：该锁是否公平锁 
- `isHeldByCurrentThread()`： 当前线程是否保持锁锁定，线程的执行lock方法的前后分别是false和true 
- `isLock()`：此锁是否有任意线程占用 
- `lockInterruptibly()`：如果当前线程未被中断，获取锁 
- `tryLock()`：尝试获得锁，仅在调用时锁未被线程占用，获得锁 
- `tryLock(long timeout TimeUnit unit)`：如果锁在给定等待时间内没有被另一个线程保持，则获取该锁。
- **`ReentrantLock`:ReentantLock继承接口Lock并实现了接口中定义的方法，他是一种可重入锁，除了能完成synchronized所能完成的所有工作外，还提供了诸如可响应中断锁、可轮询锁请求、定时锁等避免多线程死锁的方法。**


## ReentrantLock 与Synchronized

1. ReentrantLock通过方法lock()与unlock()来进行加锁与解锁操作，与synchronized会被JVM自动解锁机制不同，ReentrantLock加锁后需要手动进行解锁。为了避免程序出现异常而无法正常解锁的情况，使用ReentrantLock必须在finally控制块中进行解锁操作。 
2. ReentrantLock相比synchronized的优势是可中断、公平锁、多个锁。这种情况下需要使用ReentrantLock。

```java
public class MyService {
    private Lock lock = new ReentrantLock();
    // lock=new ReentrantLock(true);//公平锁
    // lock=new ReentrantLock(false);//非公平锁
    private Condition condition = lock.newCondition();
    public void testMethod() {
        try {
            lock.lock();
            System.out.println("开始wait");
            condition.await();

            condition.signal();
            for (int i = 0; i < 5; i++) {
                System.out.println("ThreadName=" + Thread.currentThread().getName() + (" " + (i + 1)));
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            lock.unlock();
        }
    }
}
```

1. Condition类的awiat方法和Object类的wait方法等效 
2. Condition类的signal方法和Object类的notify方法等效 
3. Condition类的signalAll方法和Object类的notifyAll方法等效 
4. ReentrantLock类可以唤醒指定条件的线程，而object的唤醒是随机的

## Semaphore信号量

<p style="text-indent:2em">
Semaphore是一种基于计数的信号量。它可以设定一个阈值，基于此，多个线程竞争获取许可信号，做完自己的申请后归还，超过阈值后，线程申请许可信号将会被阻塞。Semaphore可以用来构建一些对象池，资源池之类的，比如数据库连接池.
</p>

```java
Semaphore semp = new Semaphore(5);
try { // 申请许可
    semp.acquire();
    try { // 业务逻辑
    } catch (Exception e) {

    } finally { // 释放许可
        semp.release();
    }
} catch (InterruptedException e) {

}
```

## 原子操作

- AtomicInteger
- AtomicBoolean
- AtomicInteger
- AtomicLong
- AtomicReference<V>
- 原子操作将原本类似于`i++`这种不是原子操作的变为原子操作

```java
@Data
@AllArgsConstructor
public class User {
    private String name;
    private Integer age;
}
public static void main( String[] args ) {
    User user1 = new User("张三", 23);
    User user2 = new User("李四", 25);
    User user3 = new User("王五", 20);

	//初始化为 user1
    AtomicReference<User> atomicReference = new AtomicReference<>();
    atomicReference.set(user1);

	//把 user2 赋给 atomicReference
    atomicReference.compareAndSet(user1, user2);
    System.out.println(atomicReference.get());

	//把 user3 赋给 atomicReference
    atomicReference.compareAndSet(user1, user3);
    System.out.println(atomicReference.get());
}
```


## 公平锁、非公平锁、可重入锁

<p style="text-indent:2em">
JVM按随机、就近原则分配锁的机制则称为不公平锁，ReentrantLock在构造函数中提供了是否公平锁的初始化方式，默认为非公平锁。非公平锁实际执行的效率要远远超出公平锁，除非程序有特殊需要，否则最常用非公平锁的分配机制。加锁前检查是否有排队等待的线程，优先排队等待的线程，先来先得。
</p>

<p style="text-indent:2em">
公平锁指的是锁的分配机制是公平的，通常先对锁提出获取请求的线程会先被分配到锁，ReentrantLock在构造函数中提供了是否公平锁的初始化方式来定义公平锁。加锁时不考虑排队等待问题，直接尝试获取锁，获取不到自动到队尾等待。非公平锁性能比公平锁高5~10倍，因为公平锁需要在多核的情况下维护一个队列。Java中的synchronized是非公平锁，ReentrantLock 默认的lock()方法采用的是非公平锁。
</p>

<p style="text-indent:2em">
本文里面讲的是广义上的可重入锁，而不是单指JAVA下的ReentrantLock。可重入锁，也叫做递归锁，指的是同一线程 外层函数获得锁之后 ，内层递归函数仍然有获取该锁的代码，但不受影响。在JAVA环境下 ReentrantLock 和synchronized 都是 可重入锁。
</p>

## 共享锁、独占锁

<p style="text-indent:2em">
独占锁模式下，每次只能有一个线程能持有锁，ReentrantLock就是以独占方式实现的互斥锁。独占锁是一种悲观保守的加锁策略，它避免了读/读冲突，如果某个只读线程获取锁，则其他读线程都只能等待，这种情况下就限制了不必要的并发性，因为读操作并不会影响数据的一致性。
</p>

<p style="text-indent:2em">
共享锁则允许多个线程同时获取锁，并发访问 共享资源，如：ReadWriteLock。共享锁则是一种乐观锁，它放宽了加锁策略，允许多个执行读操作的线程同时访问共享资源。AQS的内部类Node定义了两个常量SHARED和EXCLUSIVE，他们分别标识 AQS队列中等待线程的锁获取模式。java的并发包中提供了ReadWriteLock，读-写锁。它允许一个资源可以被多个读操作访问，或者被一个 写操作访问，但两者不能同时进行。
</p>

## 偏向锁、重量级锁、轻量级锁（Synchronized的底层优化）

- 重量级锁也就是通常说synchronized的对象锁，它是通过对象的monitor进行实现的，当一个线程持有一个对象的monitor时，那么这个对象就处于锁定状态，且同时这种操作依赖操作系统，需要从用户态转到内核态，开销很大。
- 轻量级锁，使用CAS命令代替互斥量，减少了开销，适用于线程交替执行同步块，如果存在同一时间访问同一锁的情况，就会导致轻量级锁膨胀为重量级锁。轻量级锁认为竞争存在，但是竞争的程度很轻，一般两个线程对于同一个锁的操作都会错开，或者说稍微等待一下（自旋），另一个线程就会释放锁。 但是当自旋超过一定的次数，或者一个线程在持有锁，一个在自旋，又有第三个来访时，轻量级锁膨胀为重量级锁，重量级锁使除了拥有锁的线程以外的线程都阻塞，防止CPU空转。
- CAS 即 compare and swap,比较并交换，从地址 V 读取值 A，执行多步计算来获得新值 B，然后使用 CAS 将 V 的值从 A 改为 B。如果 V 处的值尚未同时更改，则 CAS 操作成功。否则，将重新进行CAS操作。存在循环操作和ABA问题。
- 偏向锁认为大多数情况下不存在多线程竞争，所以将对象一开始设置为偏向锁状态，偏向的线程访问资源时，不需要额外的操作，而非偏向线程需要判断是否竞争资源，从而升级锁。


## 锁的优化

- **减少锁持有时间**：只用在有线程安全要求的程序上加锁
- **减小锁粒度**：将大对象（这个对象可能会被很多线程访问），拆成小对象，大大增加并行度，降低锁竞争。降低了锁的竞争，偏向锁，轻量级锁成功率才会提高。最最典型的减小锁粒度的案例就是ConcurrentHashMap。
- **锁分离**：最常见的锁分离就是读写锁ReadWriteLock，根据功能进行分离成读锁和写锁，这样读读不互斥，读写互斥，写写互斥，即保证了线程安全，又提高了性能。读写分离思想可以延伸，只要操作互不影响，锁就可以分离。比如LinkedBlockingQueue 从头部取出，从尾部放数据；
- **锁粗化**：通常情况下，为了保证多线程间的有效并发，会要求每个线程持有锁的时间尽量短，即在使用完公共资源后，应该立即释放锁。但是，凡事都有一个度，如果对同一个锁不停的进行请求、同步和释放，其本身也会消耗系统宝贵的资源，反而不利于性能的优化 。
- **锁消除**：锁消除是在编译器级别的事情。在即时编译器时，如果发现不可能被共享的对象，则可以消除这些对象的锁操作，多数是因为程序员编码不规范引起。


## 阻塞队列

- ArrayBlockingQueue（公平、非公平）：数组实现的队列（FIFO）
- LinkedBlockingQueue（两个独立锁提高并发）：可以同时生产和消费
- PriorityBlockingQueue（compareTo排序实现优先）


## volatile（变量可见性、禁止重排序）

- 变量可见性：其一是保证该变量对所有线程可见，这里的可见性指的是当一个线程修改了变量的值，那么新的值对于其他线程是可以立即获取的。
- 禁止重排序：volatile 禁止了指令重排。
- 其中需要注意的是被volatile修饰的变量在单次读取操作或者写操作都是原子性的，但是类似于`i++`这种同时读取和写的操作不是原子性的。

## 线程通信常用方法

```java
public class MyData {
    private int j = 0;

    public synchronized void add() {
        j++;
        System.out.println("线程" + Thread.currentThread().getName() + "j为：" + j);
    }

    public synchronized void dec() {
        j--;
        System.out.println("线程" + Thread.currentThread().getName() + "j为：" + j);
    }

    public int getData() {
        return j;
    }
}

public class AddRunnable implements Runnable {
    MyData data;

    public AddRunnable(MyData data) {
        this.data = data;
    }

    public void run() {
        data.add();
    }
}

public class DecRunnable implements Runnable {
    MyData data;

    public DecRunnable(MyData data) {
        this.data = data;
    }

    public void run() {
        data.dec();
    }
}

public static void main(String[] args) {
    MyData data = new MyData();
    Runnable add = new AddRunnable(data);
    Runnable dec = new DecRunnable(data);
    for (int i = 0; i < 2; i++) {
        new Thread(add).start();
        new Thread(dec).start();
    }
}
```







