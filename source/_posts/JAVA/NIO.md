---
title: NIO
date: 2024-01-21 13:30:37
categories:
  - JAVA
  - NIO
tags:
  - Channel
  - Selector
  - Buffer
author: Fanrencli
---

## NIO

- NIO使用的是channel+buffer，channel是双向的，数据保存在buffer，使用时channel+buffer一起使用。
- 网络编程主要分为5种IO模型：
    - 阻塞型IO（BIO）:当请求的数据没有准备好时，则一直等待；常用于计算（CPU）密集型；
    - 非阻塞型IO（NIO）:当请求的数据没有准备好时，则返回一个错误，然后再发送请求；
    - 多路复用IO(NIO):使用一个selector线程去轮询多个socket，若存在socket准备好数据则进行处理；这样可以使用一个线程管理多个socket链接，常用于连接数较多的情况；且轮询的线程是内核执行的，所以速度很快；但是对于轮询的数量需要限制，否则程序效率下降。
    - 信号驱动IO：在发起请求时，会给对应的socket注册一个响应函数，然后继续执行其他操作，当数据准备好的时候，则调用之前注册的响应函数
    - 异步IO（AIO）：在发送请求之后，继续执行其他操作，当数据准备好的时候告诉线程，底层由内核epoll支持。
- `NIO`核心操作类主要有三个:`channel`,`Buffers`,`Selector`
    - `channel`:数据传输的通道，于buffer缓冲区结合使用，通过通道向缓冲区进行读取写入数据，类似于BIO中的Stream流，但是流是单向的，通道是双向的
        - `FileChannel`:IO文件输入输出的通道类
        - `DatagramChannel`:UDP通道实现类
        - `ServerSocketChannel`只作为判断是否有客户端连接
        - `SocketChannel`（用于客户端写数据给服务端，服务端写数据给客户端）
    - `Buffers`:用于接收通道中的数据，注意无论是客户端还是服务器端的数据都需要从通过缓冲区进入通道中
        - `ByteBuffer`
        - `CharBuffer`
        - `DoubleBuffer`
        - `FloatBuffer`
        - `IntBuffer`
        - `LongBuffer`
        - `ShortBuffer`
    - `Selector`: 用于监听缓存区是否有数据准备好

### Channel

`channel`接口主要有两个方法

```java
public interface Channel extends Closeable {

    /**
     * Tells whether or not this channel is open.
     *
     * @return {@code true} if, and only if, this channel is open
     */
    public boolean isOpen();

    /**
     * Closes this channel.
     *
     * <p> After a channel is closed, any further attempt to invoke I/O
     * operations upon it will cause a {@link ClosedChannelException} to be
     * thrown.
     *
     * <p> If this channel is already closed then invoking this method has no
     * effect.
     *
     * <p> This method may be invoked at any time.  If some other thread has
     * already invoked it, however, then another invocation will block until
     * the first invocation is complete, after which it will return without
     * effect. </p>
     *
     * @throws  IOException  If an I/O error occurs
     */
    public void close() throws IOException;

}
```

#### FileChannel

`FileChannel`主要包含以下方法
- `close()`:关闭通道
- `size()`:获取通道关联的文件的大小
- `position()`:获取通道当前关联文件的游标位置
- `position(Long pos)`:设置游标pos位置，从文件的某个文职开始读取或写入数据
- `truncate(int size)`:截取文件大小
- `force(boolean bool)`:将通道中尚未写入磁盘的数据强制写入磁盘，bool参数决定是否同时写入元数据（权限信息）
- `transferTo(position，count，tochannel)`:将数据从某个文件的通道传输到另一个文件的通道
- `transferFrom(fromchannel,position,count)`:将某个文件的通道数据传输到当前通道中,从position开始传输count个字节

读取数据

```java
RandomAccessFile accessFile = new RandomAccessFile("BIO.iml","rw");
FileChannel channel = accessFile.getChannel();
ByteBuffer allocate = ByteBuffer.allocate(1024);
// 将数据读取到缓存中,此时缓冲区的游标位置pos为数据的大小位置
int read = channel.read(allocate);
while(read!=-1){
    // 为了能够读取到最新的数据，将limit设置为pos，将pos设置为0，这样读取数据就是从pos->limit
    allocate.flip();
    while(allocate.hasRemaining()){
        System.out.print((char)allocate.get());
    }
    // 将pos置为0，limit置为容量大小
    allocate.clear();
    read = channel.read(allocate);
}
channel.close();
accessFile.close();
```

写数据

```java
RandomAccessFile accessFile = new RandomAccessFile("1.txt","rw");
FileChannel channel = accessFile.getChannel();
ByteBuffer allocate = ByteBuffer.allocate(1024);、
// 写数据之前先清空数据
allocate.clear();
allocate.put("this is test content".getBytes());
// 写入数据之后重置pos位置和linmit位置
allocate.flip();
channel.write(allocate);
channel.close();
accessFile.close();
```

文件复制

```java
 RandomAccessFile accessFile = new RandomAccessFile("1.txt","rw");
RandomAccessFile accessFile1 = new RandomAccessFile("2.txt","rw");
FileChannel channel1 = accessFile1.getChannel();
FileChannel channel = accessFile.getChannel();
channel.transferTo(0,channel.size(),channel1);
accessFile.close();
accessFile1.close();
```

#### ServerSocketChannel(TCP)

总的看来， ServerSocketChannel相当于SocketChannel的包装类，ServerSocketChannel通过监听是否有SocketChannel的连接，如果有连接则返回一个SocketChannel对象进行操作。ServerSocketChannel只作为一个包装类完成相关的配置，真正的数据操作都是SocketChannel完成。其中

```java
ServerSocketChannel serverSocketChannel = ServerSocketChannel.open();
serverSocketChannel.configureBlocking(false);
// 创建一个socket进行绑定端口
serverSocketChannel.socket().bind(new InetSocketAddress(8088));
//serverSocketChannel.bind(new InetSocketAddress(8088));
ByteBuffer byteBuffer =ByteBuffer.wrap("this is test ".getBytes());
while(true){
    System.out.println("start to wait for connection");
    SocketChannel accept = serverSocketChannel.accept();
    if (accept==null){
        System.out.println("无连接");
        Thread.sleep(200);
        continue;
    }
    System.out.println("有连接进入："+accept.getRemoteAddress());
    // 重置pos位置，不充值limit位置
    byteBuffer.rewind();
    // 向客户端写入数据
    accept.write(byteBuffer);
    // 只关闭此连接，不关闭整个监听
    accept.close();
}
serverSocketChannel.close();
```


#### SocketChannel(TCP)

```java
//        SocketChannel socketChannel = SocketChannel.open(new InetSocketAddress(8088));
SocketChannel socketChannel = SocketChannel.open();
// connect方法是客户端主动连接方法，服务端一般用bind方法进行监听客户端的连接，如果客户端使用了bind方法，那么则指定从哪个端口发出请求，如果不指定则有客户端随机选择
socketChannel.connect(new InetSocketAddress(8088));
// 连接校验
System.out.println(socketChannel.isOpen()&&socketChannel.isConnected());
socketChannel.configureBlocking(false);
socketChannel.setOption(StandardSocketOptions.SO_KEEPALIVE,true);
socketChannel.getOption(StandardSocketOptions.SO_KEEPALIVE);
// 读取数据
ByteBuffer byteBuffer = ByteBuffer.allocate(1024);
socketChannel.read(byteBuffer);
socketChannel.close();
System.out.println("ok");
```

#### DatagramChannel(UDP)

UDP不需要连接的网络通信协议，只需要知道目的地址，然后发送数据包，不关心数据是否可以正确被接受。UDP接收者只接受不同网络地址的数据包，每个包中包含了所需要的所有信息。

```java
DatagramChannel datagramChannel = DatagramChannel.open();
datagramChannel.bind(new InetSocketAddress(8088));
// 接收数据
ByteBuffer byteBuffer = ByteBuffer.allocate(1024);
byteBuffer.clear();
// addr包含了发送数据的地址和端口等信息
SocketAddress addr = datagramChannel.receive(byteBuffer);
// 发送数据
byteBuffer.clear();
byteBuffer.put("thsi is test".getBytes());
datagramChannel.send(byteBuffer,new InetSocketAddress(8089));
// UDP不存在真正意义上的连接，但是代码中可以通过connect方法向特定的地址发送和接收数据
datagramChannel.connect(new InetSocketAddress(8888));
datagramChannel.read(byteBuffer);
datagramChannel.write(byteBuffer);
```
#### 分散&聚集

```java
DatagramChannel datagramChannel = DatagramChannel.open();
datagramChannel.bind(new InetSocketAddress(8088));
// 分散，将数据读取到不同的缓存中，分别时消息头和消息体
ByteBuffer byteBuffer = ByteBuffer.allocate(128);
ByteBuffer byteBuffer2 = ByteBuffer.allocate(1024);
datagramChannel.read(new ByteBuffer[]{byteBuffer, byteBuffer2});
// 聚集，将消息头和消息体按顺序写入通道
datagramChannel.write(new ByteBuffer[]{byteBuffer,byteBuffer2});
```

### Buffer

buffer使用之前一般遵循以下步骤：
- 在写入数据之前需要`clear()`方法将游标放置于起始位置，将终点放置于容量大小的位置
- 由于在读取数据之前肯定进行过写入数据，因此需要`flip()`方法将终点放置于游标当前的位置，将游标放置于起始位置。
- `compact()`方法先判断当前数据是否已经全都读取完成，如果没有读取完，则将剩下的数据移动到最前端，并清除已经读取的数据


```java
IntBuffer intBuffer = IntBuffer.allocate(8);
for (int i = 0; i < intBuffer.capacity(); i++) {
    intBuffer.put(2*(i+1));
}
intBuffer.flip();
while (intBuffer.hasRemaining()){
    // get()方法一次读取一个字节
    System.out.println(intBuffer.get()+"->");
}
```

- Buffer中三大属性：Position,limit,Capacity

Buffer可以看作为一段连续的内存地址，在读模式中，position作为游标起始位置，指定了读取数据从哪里开始，每读取一次数据之后position都会向后移动，直到等于limit就是终点，limit指定了读到哪里是终点，capacity指定了整个缓存的大小。在写模式中，position也是作为游标指定了写入数据从哪里开始，limit也作为写数据的终点，当然一般limit等于capacity。

- 向Buffer中写入数据的两种方式：
    - 通过`put()`方法写入数据
    - 通过`channel.read(buffer)`写入数据
- 从buffer中读取数据的两种方式：
    - 通过`get()`方法读取数据
    - 通过`channel.write(buffer)`方法读取数据

- `rewind()`:将position设置为0，表示可以重新从头读取数据，或者重新从头写入数据
- `mark()`：将当前的position作一个标记，后续可以通过`reset()`方法将position重置为这个标记位置

- 缓冲区分片

```java

ByteBuffer byteBuffer = ByteBuffer.allocate(20);
for (int i = 0; i < byteBuffer.capacity(); i++) {
    byteBuffer.put((byte)i);
}
byteBuffer.position(3);
byteBuffer.limit(10);
// 分片中的数据与原缓冲区共享
ByteBuffer slice = byteBuffer.slice();
for (int i = 0; i < slice.capacity(); i++) {
    slice.put((byte)(i*10));
}
byteBuffer.position(0);
byteBuffer.limit(byteBuffer.capacity());
while(byteBuffer.hasRemaining()){
    System.out.print(byteBuffer.get()+"->");
}

```

- 只读缓冲区

```java

ByteBuffer byteBuffer = ByteBuffer.allocate(20);
for (int i = 0; i < byteBuffer.capacity(); i++) {
    byteBuffer.put((byte)i);
}
// 只读缓冲区的数据也是共享的
ByteBuffer slice = byteBuffer.asReadOnlyBuffer();
byteBuffer.clear();
for (int i = 0; i < 10; i++) {
    byteBuffer.put(i,(byte)(i*10));
}
slice.clear();
while(slice.hasRemaining()){
    System.out.print(slice.get()+"->");
}

```

- 直接缓冲区：JDK将尽量直接使用操作系统的IO操作进行读写数据更快

```java
//使用方法与其他类似
ByteBuffer byteBuffer = ByteBuffer.allocateDirect(20);
```

- 内存映射缓冲区

```java
RandomAccessFile accessFile = new RandomAccessFile("3.txt","rw");
FileChannel fc = accessFile.getChannel();
MappedByteBuffer map = fc.map(FileChannel.MapMode.READ_WRITE, 0, 128);
map.put(0,(byte)127);
map.put(1,(byte)55);
fc.close();
```

### NIO服务器端

```java
public class NIO {
    public static void main(String[] args) throws IOException{
        ServerSocketChannel serverSocketChannel = ServerSocketChannel.open();
        serverSocketChannel.configureBlocking(false);
        serverSocketChannel.bind(new InetSocketAddress(9999));
        System.out.println("NIO server has start on port " + serverSocketChannel.getLocalAddress());
        ServerSocketChannel serverSocketChannel1 = ServerSocketChannel.open();
        serverSocketChannel1.configureBlocking(false);
        serverSocketChannel1.bind(new InetSocketAddress(8888));
        System.out.println("NIO server has start on port " + serverSocketChannel1.getLocalAddress());

        Selector selector = Selector.open();
        serverSocketChannel1.register(selector,SelectionKey.OP_ACCEPT);
        serverSocketChannel.register(selector,SelectionKey.OP_ACCEPT);
        ByteBuffer byteBuffer = ByteBuffer.allocate(1024);
        while (true){
            int select = selector.select();
            if(select==0){
                continue;
            }
            Set<SelectionKey> selectionKeys = selector.selectedKeys();
            Iterator<SelectionKey> iter = selectionKeys.iterator();
            while (iter.hasNext()){
                SelectionKey selectionKey = iter.next();
                if(selectionKey.isAcceptable()){
                    ServerSocketChannel channel = (ServerSocketChannel) selectionKey.channel();
                    SocketChannel socketChannel = channel.accept();
                    System.out.println("connection from " +socketChannel.getRemoteAddress());

                    socketChannel.configureBlocking(false);
                    socketChannel.register(selector,SelectionKey.OP_READ);
                }
                if(selectionKey.isReadable()){
                    SocketChannel socketChannel = (SocketChannel) selectionKey.channel();
                    socketChannel.read(byteBuffer);
                    String request = new String(byteBuffer.array()).trim();
                    byteBuffer.clear();
                    System.out.println(String.format("From %s : %s ",socketChannel.getRemoteAddress(),request));
                    String response = "From NIOserver "+request +"\n";
                    socketChannel.write(ByteBuffer.wrap(response.getBytes()));

                }
                iter.remove();
            }
        }
    }
}
```