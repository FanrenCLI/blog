---
title: SocketIO
date: 2021-12-17 13:30:37
categories:
  - JAVA
  - BIO
  - NIO
tags:
  - Socket
  - ServerSocket
  - Channel
  - Selector
  - Buffer
author: Fanrencli
---
## BIO

- `BIO`即传统的IO操作接口
- 服务器端：`ServerSocket`
- 客户端：`Socket`


### 服务器端（ServerSocket）

- 构造方法：`public ServerSocket(int port) throws IOException`
- 监听客户端连接：`public Socket accept() throws IOException`
- 取得客户端的数据：`public OutputStream getInputStream() throws IOException`
- 向客户端发送数据`public OutputStream getOutputStream() throws IOException`

```java
public class Main {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        ServerSocket serverSocket = new ServerSocket(9999);
        System.out.println("等待用户链接-----------------");
        Socket client = serverSocket.accept();
        // 获取客户端输入的数据
        InputStream in = client.getInputStream();
        // 获取输入数据也可以通过in.read()/in.read(buffer)->new String(buffer,0,len);
        Scanner scanner = new Scanner(in);
        scanner.useDelimiter("\n");
        // 获取向客户端输出数据的流，并向客户端输出数据
        OutputStream out = client.getOutputStream();
        // 向客户端输出数据也可以用out.write("helloworld".getbytes())
        PrintStream printStream = new PrintStream(out);
        while(scanner.hasNext()){
            String req = scanner.next();
            if("q".equals(req)){
                break;
            }
            System.out.println(req);
            printStream.println(req);//注意这里的println,也可以换位printStream.print(req+"\n")否则无法输出
        }
        out.close();
        client.close();
        serverSocket.close();
    }
}
```

### 客户端（Socket）
- 构造方法：`public Socket(String IP, int port) throws IOException`
- 取得服务器的数据：`public OutputStream getInputStream() throws IOException`
- 向服务器发送数据`public OutputStream getOutputStream() throws IOException`


```java
public class test1 {
    public static void main(String[] args) throws Exception {
        Socket scoket =new Socket("localhost",9999);
        Scanner scanner = new Scanner(scoket.getInputStream());
        scanner.useDelimiter("\n");
        Scanner input = new Scanner(System.in);
        input.useDelimiter("\n");
        PrintStream printStream = new PrintStream(scoket.getOutputStream());
        while (true){
            System.out.print("Inpu data:");
            if(input.hasNext()){
                String str = input.next();
                if("q".equals(str)){
                    break;
                }
                printStream.print(str+"\n");
            }
            if(scanner.hasNext()){
                System.out.println(scanner.next());
            }
        }
        input.close();
        printStream.close();
        scanner.close();
        scoket.close();
    }
}
```

### 多线程解决BIO带来的问题

- BIO即阻塞IO，如果仅用单线程处理，那么只允许一个客户链接服务器，要实现多个客户链接，就需要采用多线程处理


```java
class test implements Runnable{
    private Socket clientsocket;
    public test(Socket socket){
        this.clientsocket = socket;
    }
    @Override
    public void run() {
        try{
            Scanner input = new Scanner(this.clientsocket.getInputStream());
            while (input.hasNext()){
                String request = input.next();
                if("quit".equals(request)){
                    break;
                }
                System.out.println(String.format("From %s : %s",this.clientsocket.getRemoteSocketAddress(),request));
                String response = "From BIOserver "+request +"\n";
                this.clientsocket.getOutputStream().write(response.getBytes());
            }
            input.close();
            this.clientsocket.close();
        }catch (Exception e){
            e.printStackTrace();
        }

    }
}
public class BIO_ThreadPool {
    public static void main(String[] args) throws IOException {
        ExecutorService executorService = Executors.newFixedThreadPool(3);
        ServerSocket serverSocket = new ServerSocket(9999);
        System.out.println("BIO server has started, listening on port"+ serverSocket.getLocalSocketAddress());
        // 循环监听是否有客户端连接，并分配线程执行
        while (true){
            Socket clientsocket = serverSocket.accept();
            executorService.submit(new test(clientsocket));
            System.out.println("connection from "+clientsocket.getRemoteSocketAddress());

        }

    }
}
```

## NIO

- 网络编程主要分为5种IO模型：
    - 阻塞型IO（BIO）:当请求的数据没有准备好时，则一直等待；常用于计算（CPU）密集型；
    - 非阻塞型IO（NIO）:当请求的数据没有准备好时，则返回一个错误，然后再发送请求；
    - 多路复用IO(NIO):使用一个selector线程去轮询多个socket，若存在socket准备好数据则进行处理；这样可以使用一个线程管理多个socket链接，常用于连接数较多的情况；且轮询的线程是内核执行的，所以速度很快；但是对于轮询的数量需要限制，否则程序效率下降。
    - 信号驱动IO：在发起请求时，会给对应的socket注册一个响应函数，然后继续执行其他操作，当数据准备好的时候，则调用之前注册的响应函数
    - 异步IO（AIO）：在发送请求之后，继续执行其他操作，当数据准备好的时候告诉线程，底层由内核epoll支持。
- `NIO`所涉及的相关操作类
    - `ServerSocketChannel`只作为判断是否有客户端连接:
        - 创建一个服务器通道 `public static ServerSocketChannel open() throws IOException`;
        - 绑定通道对应的端口：`public final ServerSocketChannel bind(SocketAddress local) throws OException`;
        - 将通道注册到selector中：`public final SelectionKey register(Selector sel, int ops)        throws ClosedChannelException`
    - `SocketChannel`（用于客户端写数据给服务端，服务端写数据给客户端）:
        - 创建一个服务器通道 `public static SocketChannel open() throws IOException`;
        - 链接服务器：`public abstract boolean connect(SocketAddress remote) throws IOException;`;
    - `ByteBuffer`
        - 用于接收通道中的数据，注意无论是客户端还是服务器端的数据都需要从通过缓冲区进入通道中
    - `Selector`
        - 创建一个selector：`public static Selector open() throws IOException`;

### NIO客户端

```java
public class NIO {
    public static void main(String[] args) throws IOException{

        SocketChannel socketChannel = SocketChannel.open(new InetSocketAddress("www.baidu.com", 80));
        socketChannel.configureBlocking(false);
        ByteBuffer byteBuffer = ByteBuffer.allocate(1024);
        socketChannel.read(byteBuffer); //读取客户端通道获取的数据
        // socketChannel.write(byteBuffer); 向客户端通道写数据
        socketChannel.close();
        System.out.println("test end!");
        System.out.println(new String(byteBuffer.array()));
    }
}
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