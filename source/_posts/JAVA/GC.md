---
title: JVM
date: 2021-12-20 17:26:18
categories:
  - JAVA
tags:
  - JVM
  - GC
author: Fanrencli
---

## JVM相关知识

### 类的加载


1. 加载

    类的加载阶段主要是从文件系统或者网络中加载Class文件，class文件在文件开头有magic标识。在类的加载阶段只关注是否class文件的加载，对加载到的class文件是合法并不校验。其中类加载器主要分类两类：
    - 启动类加载器（Bootstrap Classloader）
        主要用于加载java核心类（String、Class)等一些sun包下的类和javax包下的。
    - 自定义加载器（继承了ClassLoad的加载器）
        扩展类加载器和应用加载器都是自定义加载器

    此外，启动类加载器和扩展类加载器以及应用加载器是符合双亲委派机制，但是他们不是继承关系，只是在加载类的过程中存在向上寻找的过程。因此，如果用户自定义加载器时，也可以通过实现findclass()方法实现双亲委派的过程。双亲委派机制的优势
    - 避免类的重复加载
    - 防止核心Api被篡改


2. 链接
    - 验证

        验证阶段主要用于对加载得到的文件内容进行校验，是否符合jVM的规范。主要为文件格式验证，元数据验证，字节码验证，符号引用验证
    - 准备

        在这个步骤为类变量（Static）分配内存并设置初始的默认值,注意这里不包含static final的变量，这类变量在加载的时候就进行了分配，实例变量也会分配内存，但是是在堆中分配
    - 解析

        主要将符号引用转为直接引用


3. 初始化阶段
    - 执行类构造器方法<clinit>，不是类的构造器。这个方法是这个类中所有类变量赋值动作和静态代码块的集合，

### 虚拟机栈


虚拟机栈为每个线程私有，不同的线程都有自己的虚拟机栈，栈中主要以栈帧为单位存放数据，每一个栈帧对应一个方法。栈帧中主要存放局部变量表、操作数栈、动态链接、方法返回地址、一些附加的信息。
当一个方法调用另一个方法则会在栈中入栈一个栈帧，返回一个方法则销毁一个栈帧。

- 局部变量表：这是垃圾收集判断是否对象已经死亡的主要地方。方法中所有的局部变量都存放在这里，可以存储局部变量的数据
- 操作数栈：主要是通过一个数组进行实现，通过push和pop对数据进行入栈和出栈。作为运算过程中的临时存储空间。
- 动态链接：动态链接就是符号引用，通过符号引用可以定位运行时常量池中的方法，从而在替换为直接引用
- 方法的返回地址：方法的返回地址主要用于将返回参数压入下一个方法的栈中，存放的是程序计数器的数据，用于确定下一个运行的指令地址。

### 对象

对象包含：
- 对象头：分为Mark word（8字节）和类元信息（8字节）
  - Mark word中当对象处于无锁状态时如果没有调用过hashcode方法，那么对象头中不会生成hashcode,只要对象头中没有hashcode,那么在锁升级的过程中就可以升级偏向锁，但是如果调用过hashcode方法，那么对象头中就会生成hasdcode，此时如果遇到锁升级就会直接跳过偏向锁，直接升级为轻量级锁。此时hashcode会被存储在对应的monitor中。
  - 在多线程并发的过程中，获取对象的锁就是根据对象头中记录的monitor地址，获取对象的monitor，并在monitor中记录对应的信息
- 实例数据：类的成员变量
- 对齐填充：补齐为8字节的倍数
其中mark word包含对象的hashcode，分代年龄，是否偏向锁，锁标志位，偏向锁的线程

## JVM 内存模型

![JVM内存模型](http://fanrencli.cn/fanrencli.cn/20200101151338500.png)

- 堆：（new）对象存储，数组（在jdk7后字符串常量、静态变量）；
- 方法区：类的加载信息、常量、即时编译后的代码；
- 程序计数器：每个线程在这里都会私有一个标记代码的运行位置；
- 虚拟机栈：存储每个方法运行创建的栈帧（局部变量表(对象的引用（对C++中的指针的封装）、基础数据类型)、操作数栈、动态链接、方法出口）；
- 本地方法栈：存储本地方法的服务；
- 版本变化：Java6之前，常量池是存放在方法区（永久代）中的。Java7中将常量池是存放到了堆中。 Java8之后，取消了整个永久代区域，取而代之的是元空间。运行时常量池和静态常量池存放在元空间中，而字符串常量池依然存放在堆中。
## GC算法

- 标记-清除：先标记，然后清除，但是会导致内存碎片化
- 标记-复制：将内存划分为两块，每次只使用一块，一块使用完后，将清除垃圾后剩下的对象复制到另一块上，然后整体清除此内存块
- 标记-整理：先标记，将活着的对象移动向一端，然后按照边界进行清除
- 分代收集：将内存划分为年轻代和老年代，不同的区域选择不同的算法，一般年轻代选择复制，老年代选择清除或者整理

### HotSpot JVM

作为主流的JVM，HotSpot虚拟机采用分代收集的方法，分为老年代和新生代。其中新生代使用标记-复制方法进行垃圾回收，从而将新生代分为Eden和Survivor1和Survivor0三个部分，其中Eden占比较大，如果Eden满了就进行minorGC，将活着的对象放入Survivor中空着的区域，清空Eden和另一个Survivor区域。大对象可能直接进入老年代。
- minorGC：Eden满了会触发minorGC,Survivor满了不会触发，收集整个新生代的垃圾，会触发STW。
- majorGC:收集整个老年代的垃圾，老年代空间不足就会触发
- FullGC：收集整个java堆和方法区的垃圾，fullgc只是一个概念，意指所有内存空间都会进行垃圾回收，但是需要结合具体的垃圾收集器进行分析。

![JVM对象分配过程](http://fanrencli.cn/fanrencli.cn/jvm_pic1.jpg)


对象的分配不一定都在堆上分配，首先通过逃逸分析，如果这个对象只在此方法中使用，则认为没有逃逸，就在栈中分配内存，随着方法销毁。
TLAB的出现是由于堆内空间共享，如果多个线程同时创建对象申请空间就会存在竞争，此时通过给每个线程分配一个TLAB的空间，这样就不需要竞争空间了，TLAB在空间上是私有的，但是内部的对象是共享的。
### 确定垃圾

- 引用计数法：如果一个对象没有一个与之相关的引用，那么他的引用计数都为0，此时可以当作垃圾进行回收
- 可达性分析：通过`GC root`（虚拟机栈中的对象引用、方法区中类静态属性引用的对象、方法区中常量引用的对象、本地方法栈中JNI引用的对象）对对象进行分析，如果通过`GC root`可以找到的对象则认为此对象活着，否则作为垃圾；

### 标记过程

- 其中标记分为两次标记：通过GCroot节点第一次的检查可以发现不能到达的对象，然后对这些对象进行第一次标记，在标记过程对这些对象进行检查——是否重写了finalize方法、或者finalize方法是否被调用过，结果分为两种情况：
    - 若重写了finalize方法且这个方法没有被调用过则对这个方法进行调用
    - 若没有重写则不进行调用。
    - 若重写了finalize方法但是之前调用过此方法则不执行。
- 通过第一次标记之后调用finalize方法有的对象可能又被重新引用（逃离死亡），而有的对象则没有逃离，所以在GC发起第二次标记的时候剩下的对象则被清除。其中要注意的是在执行finalize方法时，GC不会等待finalize方法，主要是因为finalize方法可能会长时间执行或假死而导致整个系统的崩溃。

### 标记产生的相关问题解析

在GC过程中要对对象进行标记，在此过程中对象不能再进行更改引用，因此在GC过程中必须要暂停所有线程，但是暂停不能过于频繁，也不能太少，要选择合适的点进行暂停（安全点，这个安全点一般在需要长时间执行的代码处进行标记（for循环、方法调用、异常跳转）），在运行到安全点暂停之后进行GC的检查过程中如果对所有的对象进行遍历检查， 代价过高，hotspot中采用oop数据结构对GCroot节点中的对象引用进行标记，在检查时就可以很快的找到引用对象的位置，因而可以快速确定未被引用的对象位置，但是在代码运行过程中对象的引用是不断变化的，可能运行到这行代码对象引用还是这样，但是下一行代码又产生新的对象引用，这样oop的内容过多，导致一系列的问题，所以在代码运行过程中只在安全点处进行标记——运行到这行代码时，那些对象有引用，那些对象没有引用？。

上面说到要对所有线程进行暂停，但是有的线程执行时，不能立即暂停，需要让它运行到最近的安全点然后暂停，对于所有线程如何暂停——当需要暂停时，JVM生成一个test轮询指令，所有线程对这个指令进行轮询，当线程轮询到这个指令时就暂停。然而还有一个问题，就是在进行GC需要暂停时，在运行的线程可以进行轮询然后暂停，但是若线程此时处于sleep或者blocked状态时，显然它收不到轮询的指令，但是JVM又不知道这个线程什么时候会开始执行，所以为了防止在暂停时由于sleep或blocked状态的线程开始运行而导致对象引用发生变化，JVM设置一个安全域（safeRegion）——在安全域中的线程禁止对象引用发生改变。在线程要离开安全域时检查系统是否完成根节点枚举或者整个GC过程，如果完成则继续运行，反之则等待。

### 垃圾收集算法具体实现（垃圾收集器）

- Serial垃圾收集器（单线程、复制算法）
- Serial Old收集器（单线程标记整理算法）
- ParNew垃圾收集器（Serial+多线程）
- Parallel Scavenge收集器（多线程复制算法、高效）
- Parallel Old收集器（多线程标记整理算法）
- CMS收集器（多线程标记清除算法）
- G1收集器

#### JDK垃圾收集器详解

JDK（Java Development Kit）提供了多种垃圾收集器（Garbage Collector, GC），每种垃圾收集器都有其特定的使用场景和优缺点。以下是JDK中常见的垃圾收集器及其详细介绍：

---

#### 1. **Serial收集器**
- **特点**：单线程收集器，使用一个线程进行垃圾回收，工作时会暂停所有应用线程（Stop-The-World）。
- **适用场景**：适用于单核CPU或小型应用，内存占用较小。
- **启用方式**：`-XX:+UseSerialGC`
- **优点**：简单高效，适合客户端应用。
- **缺点**：停顿时间较长，不适合大内存或对响应时间敏感的应用。

---

#### 2. **Parallel收集器（Throughput收集器）**
- **特点**：多线程收集器，使用多个线程进行垃圾回收，适合多核CPU。
- **适用场景**：适用于多核CPU、大内存、对吞吐量要求高的应用。
- **启用方式**：`-XX:+UseParallelGC`
- **优点**：吞吐量高，适合后台计算任务。
- **缺点**：停顿时间较长，不适合对响应时间敏感的应用。

---

#### 3. **Parallel Old收集器**
- **特点**：Parallel收集器的老年代版本，使用多线程进行老年代的垃圾回收。
- **适用场景**：与Parallel收集器配合使用，适用于多核CPU、大内存、对吞吐量要求高的应用。
- **启用方式**：`-XX:+UseParallelOldGC`
- **优点**：吞吐量高，适合后台计算任务。
- **缺点**：停顿时间较长，不适合对响应时间敏感的应用。

---

#### 4. **CMS收集器（Concurrent Mark-Sweep）**
- **特点**：并发收集器，尽量减少应用停顿时间，适合对响应时间敏感的应用。
- **适用场景**：适用于多核CPU、大内存、对响应时间敏感的应用。
- **启用方式**：`-XX:+UseConcMarkSweepGC`
- **优点**：停顿时间短，适合交互式应用。
- **缺点**：吞吐量较低，内存碎片问题较严重。

---

#### 5. **G1收集器（Garbage-First）**
- **特点**：面向服务端应用的垃圾收集器，将堆内存划分为多个区域（Region），优先回收垃圾最多的区域。
- **适用场景**：适用于大内存、多核CPU、对响应时间和吞吐量都有要求的应用。
- **启用方式**：`-XX:+UseG1GC`
- **优点**：停顿时间可控，适合大内存应用。
- **缺点**：实现复杂，内存占用较高。

---

#### 6. **ZGC收集器（Z Garbage Collector）**
- **特点**：低延迟垃圾收集器，目标是将停顿时间控制在10ms以内，适合超大内存应用。
- **适用场景**：适用于超大内存（TB级别）、对停顿时间极度敏感的应用。
- **启用方式**：`-XX:+UseZGC`
- **优点**：停顿时间极短，适合实时应用。
- **缺点**：内存占用较高，JDK 11及以上版本支持。

---

#### 7. **Shenandoah收集器**
- **特点**：低延迟垃圾收集器，与ZGC类似，目标是将停顿时间控制在10ms以内。
- **适用场景**：适用于大内存、对停顿时间敏感的应用。
- **启用方式**：`-XX:+UseShenandoahGC`
- **优点**：停顿时间短，适合实时应用。
- **缺点**：内存占用较高，JDK 12及以上版本支持。

---

#### 8. **Epsilon收集器**
- **特点**：无操作垃圾收集器，不进行任何垃圾回收，适用于性能测试和短生命周期应用。
- **适用场景**：适用于性能测试、短生命周期应用或已知内存足够的场景。
- **启用方式**：`-XX:+UseEpsilonGC`
- **优点**：无垃圾回收开销，适合特定场景。
- **缺点**：不进行垃圾回收，内存耗尽时应用会崩溃。

---

## 总结
- **Serial收集器**：适合单核CPU和小型应用。
- **Parallel收集器**：适合多核CPU和对吞吐量要求高的应用。
- **CMS收集器**：适合对响应时间敏感的应用。
- **G1收集器**：适合大内存和对响应时间、吞吐量都有要求的应用。
- **ZGC和Shenandoah收集器**：适合超大内存和对停顿时间极度敏感的应用。
- **Epsilon收集器**：适合性能测试和短生命周期应用。

---

### 四种引用类型

- 强引用：如果一个对象与GC Roots之间存在强引用，则称这个对象为强可达对象，例如：String asd = new String("");
- 软引用：软引用是使用SoftReference创建的引用，强度弱于强引用，被其引用的对象在内存不足的时候会被回收，不会产生内存溢出。

```java
  String s = new String("AABB");    // 创建强引用与String对象关联，现在该String对象为强可达状态
  SoftReference<String> softRef = new SoftReference<String>(s);     // 再创建一个软引用关联该对象
  s = null;        // 消除强引用，现在只剩下软引用与其关联，该String对象为软可达状态
  s = softRef.get();  // 重新关联上强引用
```

- 弱引用：在发生GC时，只要发现弱引用，不管系统堆空间是否足够，都会将对象进行回收。

```java
  String s = new String("Frank");    
  WeakReference<String> weakRef = new WeakReference<String>(s);
  s = null;

```

- 虚引用：当垃圾回收器准备回收一个对象时，如果发现它还有虚引用，就会在垃圾回收后，将这个虚引用加入引用队列，在其关联的虚引用出队前，不会彻底销毁该对象。 所以可以通过检查引用队列中是否有相应的虚引用来判断对象是否已经被回收了。

```java
 private static final ReferenceQueue<Student> QUEUE = new ReferenceQueue<>();
 PhantomReference<Student> phantomReference = new PhantomReference<>(new Student(), QUEUE);
 System.out.println(phantomReference.get());
```
## JVM启动参数介绍

常用的JVM参数设置：
- -XX:+PrintFlagsInitial:查看所有的参数的默认初始值
![J1](http://fanrencli.cn/fanrencli.cn/image-10.png)
- -XX:PrintFlagsFinal:查看所有的参数的最终值
- -Xms:初始堆空间大小（默认1/64)
- -Xmx:最大对空间大小（默认1/4）
- -Xmn:设置新生代的大小（初始值及最大值）
- -XX:NewRaito:配置新生代和老年代再对结构的占比
- -XX:SurvivorRatio：设置新生代中的Eden和S0、S1空间的比例
- -XX:MaxTenuringThreshold:设置新生代垃圾的最大年龄
- -XX:+PrintGC：打印GC简要信息
- -XX:HandlePromotionFailure:是否设置空间分配担保

设置GC日志参数：
- -XX:+PrintGCDetails：打印GC时的详细信息
- -XX:+PrintGCDateStamps：打印GC系统时间
- -Xloggc:<GC文件路径>:将日志存储到文件中，文件按时间取名
- -XX:+UseGCLogFileRotation：打开GC日志滚动记录功能
- -XX:NumberOfGCLogFiles：GC日志数量
- -XX:GCLogFileSize：GC日志文件大小

GC日志示例：

```txt
OpenJDK 64-Bit Server VM (25.312-b07) for linux-amd64 JRE (1.8.0_312-b07), built on Nov 13 2021 08:26:07 by "mockbuild" with gcc 8.5.0 20210514 (Red Hat 8.5.0-4)
Memory: 4k page, physical 12979592k(10600960k free), swap 4194304k(4194304k free)
CommandLine flags: -XX:GCLogFileSize=8192 -XX:InitialHeapSize=207673472 -XX:MaxHeapSize=3322775552 -XX:NumberOfGCLogFiles=5 -XX:+PrintGC -XX:+PrintGCDateStamps -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+UseCompressedClassPointers -XX:+UseCompressedOops -XX:+UseGCLogFileRotation -XX:+UseParallelGC 
2022-04-14T07:52:06.521+0000: 0.073: [GC (Allocation Failure) [PSYoungGen: 51712K->8190K(59904K)] 51712K->34086K(196608K), 0.0534755 secs] [Times: user=0.46 sys=0.21, real=0.05 secs] 
2022-04-14T07:52:06.855+0000: 0.406: [Full GC (Ergonomics) [PSYoungGen: 8176K->3053K(111616K)] [ParOldGen: 164547K->168844K(363008K)] 172723K->171898K(474624K), [Metaspace: 2946K->2946K(1056768K)], 1.3264231 secs] [Times: user=14.87 sys=0.20, real=1.33 secs] 
Heap
 PSYoungGen      total 721408K, used 7219K [0x000000077df80000, 0x00000007c0000000, 0x00000007c0000000)
  eden space 360960K, 2% used [0x000000077df80000,0x000000077e68ce50,0x0000000794000000)
  from space 360448K, 0% used [0x0000000794000000,0x0000000794000000,0x00000007aa000000)
  to   space 360448K, 0% used [0x00000007aa000000,0x00000007aa000000,0x00000007c0000000)
 ParOldGen       total 1561600K, used 765K [0x00000006f9e00000, 0x0000000759300000, 0x000000077df80000)
  object space 1561600K, 0% used [0x00000006f9e00000,0x00000006f9ebf6d8,0x0000000759300000)
 Metaspace       used 2982K, capacity 4486K, committed 4864K, reserved 1056768K
  class space    used 281K, capacity 386K, committed 512K, reserved 1048576K
```

生成DUMP文件常用参数：
- -XX:+HeapDumpOnOutOfMemoryError
- -XX:HeapDumpPath=<dump文件路径>

## JVM调优命令

- javac:编译*.java后缀的文件为class文件
  - `javac -g test.java`:`-g`表示生成的文件中包含局部变量表
- javap:反编译class文件
  - `javap <options> <classes>`
  - `javap -public test.class`:显示公共的信息
  - `javap -p test.class`:显示高于私有的信息
  - `javap -c test.class`:对代码进行反汇编
  - `javap -v test.class`:输出所有附加信息（常用）


- jps:查询当前系统中正在运行的java进程，和任务管理器中的进程ID是一致的
  - `jps -q`：只展示进程号
  - `jps -l`：展示进程对应的启动类的全路径名
  - `jps -m`：展示java进程启动时传递给main函数的参数
  - `jps -v`：展示java进程启动时包含的JVM配置参数：-Xms/-Xmx
  - 注意，如果java进程启动时使用了参数：-XX:-UserPerfData,那么jps/jstat命令将无法看到此进程
  - 此外，还可以在jps命令最后加上IP:port来收集远程主机的信息，但是必须安装jstatd

- jstat:查看JVM统计信息
  - `jstat -<options> [-t] [-h<lines>] <vmid> [<interval> [<count>]]`
  - `options`: 参数可选项有多种，其中包括：-class/-compiler/-printcompilation/-gc
  - `jstat -class <PID>`:*load*加载的类数量和字节数，*unload*卸载的类的数量和字节数，*Time*花费的时间
  ![P1](http://fanrencli.cn/fanrencli.cn/jps1.png)
  - `jstat -class <PID> <interval>`:每隔多少毫秒打印一次
  ![P2](http://fanrencli.cn/fanrencli.cn/image.png)
  - `jstat -class <PID> <interval> <count>`:每隔多少毫秒打印一次,一共打印多少次
  ![P3](http://fanrencli.cn/fanrencli.cn/image-1.png)
  - `jstat -class -t <PID>`:新增时间列，表示程序启动到输出信息的总时间
  ![P4](http://fanrencli.cn/fanrencli.cn/image-2.png)
  - `jstat -class -t -h3 <PID> <interval> <count>`:周期性输出信息时每个3行打印一行表头
  ![P5](http://fanrencli.cn/fanrencli.cn/image-3.png)
  - `jstat -compiler <PID>`:程序启动JIT编译的数量，失败的数量，不合法的数量，耗时
  ![P6](http://fanrencli.cn/fanrencli.cn/image-4.png)
  - `jstat -printcompilation <PID>`:输出已经被JIT编译的方法
  - `jstat -gc <PID>`:输出当前内存空间的使用情况：EC伊甸园区容量/EU伊甸园区使用量/S0C幸存者1容量/S0U幸存者1使用量/S1C/S1U/OC老年代容量/OU老年代使用量/MC方法区容量/MU方法区使用量/YGC年轻代GC次数/YGCT年轻代GC耗时/FGC老年代GC次数/FGCT老年代GC时间/GCT所有的GC耗时
  ![P7](http://fanrencli.cn/fanrencli.cn/image-5.png)
  - `jstat -gcutil -t -h10 <PID> 5000`:主要关注各个区域的使用占比
  ![P8](http://fanrencli.cn/fanrencli.cn/image-6.png)

- jinfo:实时查看和修改JVM参数
  - `jinfo -sysprops <PID>`:查看所有的系统属性
  ![P9](http://fanrencli.cn/fanrencli.cn/image-8.png)
  - `jinfo -flags <PID>`:查看所有被赋值的参数
  ![P10](http://fanrencli.cn/fanrencli.cn/image-7.png)
  - `jinfo -flag 具体的参数 <PID>`:查看某个具体参数的值
  ![P11](http://fanrencli.cn/fanrencli.cn/image-9.png)
  - `jinfo -flag +/-具体的参数 <PID>`:修改某个具体的参数（只有部分参数可以修改manageable）
  - `jinfo -flag 具体的参数=xxx <PID>`:修改某个具体的参数（只有部分参数可以修改manageable）

- jmap:导出内存印象文件
  - `jmap [option] <PID>`
  - `jmap -dump:format=b,file=/home/test.prof <PID>`:*format=b*标准格式
  - `jmap -dump:live,format=b,file=/home/test.prof <PID>`:*live*保存存活的对象
  - `jmap -heap <PID>`:查看堆内存的相关配置和使用情况，与jstat相似
  - `jmap -histo <PID>`:查看对象的内存占用情况

- jhat:分析jmap导出的内存文件（JDK8之后就被删除了，官方推荐使用Visualvm代替
  - `jhat <文件地址>`：分析dump文件，并在7000端口提供浏览器访问入口

- jstack:获取当前进程中的线程相关信息
  - `jstack <PID>`
  ![P12](http://fanrencli.cn/fanrencli.cn/image-11.png)


### Arthas

#### 安装


```sh
# 下载地址
curl -O https://arthas.aliyun.com/arthas-boot.jar
# 启动命令
java -jar arthas-boot.jar
# 日志路径
cd ~/logs/arthas/
```

#### 操作命令

启动完arthas后，可以选择进行进行监控，监控成功后会进入arthas的命令行界面，可以通过以下命令操作

- quit/stop:其中quit命令会退出当前的客户端对其他客户端没有影响，stop则是关闭所有客户端

- dashboard：查看当前进程的状态
  - `-i 1000`: 间隔1000ms打印一次
  - `-n 4`:总共打印4次

```sh
$ dashboard
ID     NAME                   GROUP          PRIORI STATE  %CPU    TIME   INTERRU DAEMON
17     pool-2-thread-1        system         5      WAITIN 67      0:0    false   false
27     Timer-for-arthas-dashb system         10     RUNNAB 32      0:0    false   true
11     AsyncAppender-Worker-a system         9      WAITIN 0       0:0    false   true
9      Attach Listener        system         9      RUNNAB 0       0:0    false   true
3      Finalizer              system         8      WAITIN 0       0:0    false   true
2      Reference Handler      system         10     WAITIN 0       0:0    false   true
4      Signal Dispatcher      system         9      RUNNAB 0       0:0    false   true
26     as-command-execute-dae system         10     TIMED_ 0       0:0    false   true
13     job-timeout            system         9      TIMED_ 0       0:0    false   true
1      main                   main           5      TIMED_ 0       0:0    false   false
14     nioEventLoopGroup-2-1  system         10     RUNNAB 0       0:0    false   false
18     nioEventLoopGroup-2-2  system         10     RUNNAB 0       0:0    false   false
23     nioEventLoopGroup-2-3  system         10     RUNNAB 0       0:0    false   false
15     nioEventLoopGroup-3-1  system         10     RUNNAB 0       0:0    false   false
Memory             used   total max    usage GC
heap               32M    155M  1820M  1.77% gc.ps_scavenge.count  4
ps_eden_space      14M    65M   672M   2.21% gc.ps_scavenge.time(m 166
ps_survivor_space  4M     5M    5M           s)
ps_old_gen         12M    85M   1365M  0.91% gc.ps_marksweep.count 0
nonheap            20M    23M   -1           gc.ps_marksweep.time( 0
code_cache         3M     5M    240M   1.32% ms)
Runtime
os.name                Mac OS X
os.version             10.13.4
java.version           1.8.0_162
java.home              /Library/Java/JavaVir
                       tualMachines/jdk1.8.0
                       _162.jdk/Contents/Hom
                       e/jre

```

- thread:打印当前运行的所有线程
  - `-n 2`:cpu使用前二的线程
  - `-i 5000`:统计5秒内cpu的使用率
  - `-b`：查询被阻塞的线程
- sysprop:查询或修改jvm的系统属性*sysprop [java.version]*
- sysenv:查看jvm的环境变量*sysenv [java.version]*
- heapdump:生成dump文件*heapdump testdump.hprof*
- sc:查看jvm已经加载的类的信息*sc -d com.example.test.fanren*
- sm:查看jvm已经加载的类方法的信息*sm -d com.example.test.fanren toString*
- jad:反编译类或方法的代码*sm com.example.test.fanren*
- mc:编译java文件为class文件*mc -c[指定classloader] 327a647b /tmp/Test.java*
- redefine:用本地的class文件替换已经加载的类（不可以新增属性和方法，如果某个方法正在运行，此时替换不会生效）
  - jad 命令反编译，然后可以用其它编译器，比如 vim 来修改源码
  - mc 命令来内存编译修改过的代码
  - 用 redefine 命令加载新的字节码

```shell
redefine /tmp/Test.class
redefine -c 327a647b /tmp/Test.class /tmp/Test\$Inner.class
redefine --classLoaderClass sun.misc.Launcher$AppClassLoader /tmp/Test.class /tmp/Test\$Inner.class
# 实战操作
jad --source-only com.example.demo.arthas.user.UserController > /tmp/UserController.java
mc /tmp/UserController.java -d /tmp
redefine /tmp/com/example/demo/arthas/user/UserController.class
```

以下是针对方法的命令

- monitor:监控方法被调用的次数，失败成功等相关信息，*monitor -c 5 com.fanren.test main*
- watch:查看方法被调用的返回参数*watch demo.MathGame primeFactors -x 2*
- trace:跟踪方法的调用路径*trace demo.MathGame run*
- stack:跟踪方法栈中的调用路径*stack demo.MathGame primeFactors*
- tt:查看方法被调用的耗时*tt -t -n 5 com.fanren.test main*
- profiler:生成火焰图

```sh
# 启动profiler
profiler start
# 获取已经采集的数量
profiler getSamples
# 停止，停止之后就会生成文件
profiler stop --format html
```