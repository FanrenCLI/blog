---
title: Netty
date: 2021-12-25 09:45:49
categories:
  - JAVA
  - Netty
tags:
  - pipeline
author: Fanrencli
---

## Netty

<p style="text-indent:2em">
Netty是对Java NIO的封装实现，通过Netty我们可以用简介的代码实现JavaNIO的数据通信，此处要结合之前的ScoketIO文章进行结合阅读，在本文中仅进行了Netty实现的代码进行了实现，同时需要了解BIO和NIO原生的代码实现需要在前文中阅读。
</p>

### 客户端

```java
EventLoopGroup group = new NioEventLoopGroup(); // 创建一个线程池
try {
    Bootstrap client = new Bootstrap(); // 创建客户端处理程序
    client.group(group).channel(NioSocketChannel.class)
            .option(ChannelOption.TCP_NODELAY, true) // 允许接收大块的返回数据
            .handler(new ChannelInitializer<SocketChannel>() {
                @Override
                protected void initChannel(SocketChannel socketChannel) throws Exception {
                    socketChannel.pipeline().addLast(new LengthFieldBasedFrameDecoder(65536, 0, 4, 0, 4));
                    socketChannel.pipeline().addLast(new JSONDecoder());
                    socketChannel.pipeline().addLast(new LengthFieldPrepender(4));
                    socketChannel.pipeline().addLast(new JSONEncoder());
                    socketChannel.pipeline().addLast(new EchoClientHandler()); // 追加了处理器
                }
            });
    ChannelFuture channelFuture = client.connect(HostInfo.HOST_NAME, HostInfo.PORT).sync();
    channelFuture.channel().closeFuture().sync(); // 关闭连接
} finally {
    group.shutdownGracefully();
}

```

### 服务器端

```java
EventLoopGroup bossGroup = new NioEventLoopGroup(10); // 创建接收线程池
EventLoopGroup workerGroup = new NioEventLoopGroup(20); // 创建工作线程池
System.out.println("服务器启动成功，监听端口为：" + HostInfo.PORT);
try {
    // 创建一个服务器端的程序类进行NIO启动，同时可以设置Channel
    ServerBootstrap serverBootstrap = new ServerBootstrap();   // 服务器端
    // 设置要使用的线程池以及当前的Channel类型
    serverBootstrap.group(bossGroup, workerGroup).channel(NioServerSocketChannel.class);
    // 接收到信息之后需要进行处理，于是定义子处理器
    serverBootstrap.childHandler(new ChannelInitializer<SocketChannel>() {
        @Override
        protected void initChannel(SocketChannel socketChannel) throws Exception {
            socketChannel.pipeline().addLast(new LengthFieldBasedFrameDecoder(65536,0,4,0,4)) ;
            socketChannel.pipeline().addLast(new JSONDecoder()) ;
            socketChannel.pipeline().addLast(new LengthFieldPrepender(4)) ;
            socketChannel.pipeline().addLast(new JSONEncoder()) ;
            socketChannel.pipeline().addLast(new EchoServerHandler()); // 追加了处理器
        }
    });
    // 可以直接利用常亮进行TCP协议的相关配置
    serverBootstrap.option(ChannelOption.SO_BACKLOG, 128);
    serverBootstrap.childOption(ChannelOption.SO_KEEPALIVE, true);
    // ChannelFuture描述的时异步回调的处理操作
    ChannelFuture future = serverBootstrap.bind(HostInfo.PORT).sync();
    future.channel().closeFuture().sync();// 等待Socket被关闭
} finally {
    workerGroup.shutdownGracefully() ;
    bossGroup.shutdownGracefully() ;
}
```

### Pipeline 详解

- `Pipeline`是Netty对数据处理流程的核心操作类，通过前文的一系列配置，最终操作的实现都需要在其中实现。`socketChannel.pipeline().addLast()`等一系列的方法就是添加个人的数据操作。通过Netty自定义的操作顺序，对输入输出的数据进行拆包/封装/编码/解码/自定义数据操作方法。
- `pipeline` 常用流程：数据输入->拆包->解码->相关的处理操作（继承了ChannelInboundHandlerAdapter的Handler，按照定义的顺序执行）->相关的处理操作（继承了ChannelOutboundHandlerAdapter的Handler，按照定义的顺序执行）->编码->封装->数据输出。
- 在`pipeline`是一个典型的双向链表结构，根据定义时的顺序和结构会将处理操作进行排序，有数据输入时，只执行数据输入相关的操作，数据输出时只执行数据输出的相关操作。其中数据通过ctx上下文进行传输，通过以object进行封装。
- 其中 ctx.writeAndFlush 和ctx.channel.writeAndFlush 是数据输出的信号发送源头，区别在于前者将此时的handler作为最后的handler并把数据进行编码和封装传输出去，而后者会从tail将所有的handler执行一遍后发出去。


### HTTP实现代码

```java

public class HttpServer {
    static {
        DiskFileUpload.baseDirectory = System.getProperty("user.dir") + "/upload/" ;
    }
    public void run() throws Exception {
        // 线程池是提升服务器性能的重要技术手段，利用定长的线程池可以保证核心线程的有效数量
        // 在Netty之中线程池的实现分为两类：主线程池（接收客户端连接）、工作线程池（处理客户端连接）
        EventLoopGroup bossGroup = new NioEventLoopGroup(10); // 创建接收线程池
        EventLoopGroup workerGroup = new NioEventLoopGroup(20); // 创建工作线程池
        System.out.println("服务器启动成功，监听端口为：" + HostInfo.PORT);
        try {
            // 创建一个服务器端的程序类进行NIO启动，同时可以设置Channel
            ServerBootstrap serverBootstrap = new ServerBootstrap();   // 服务器端
            // 设置要使用的线程池以及当前的Channel类型
            serverBootstrap.group(bossGroup, workerGroup).channel(NioServerSocketChannel.class);
            // 接收到信息之后需要进行处理，于是定义子处理器
            serverBootstrap.childHandler(new ChannelInitializer<SocketChannel>() {
                @Override
                protected void initChannel(SocketChannel socketChannel) throws Exception {
                    socketChannel.pipeline().addLast(new HttpResponseEncoder()) ;   // 响应编码
                    socketChannel.pipeline().addLast(new HttpRequestDecoder()) ;    // 请求解码
                    socketChannel.pipeline().addLast(new ChunkedWriteHandler()) ; // 图片传输处理器
                    socketChannel.pipeline().addLast(new HttpServerHandler()) ;
                }
            });
            // 可以直接利用常亮进行TCP协议的相关配置
            serverBootstrap.option(ChannelOption.SO_BACKLOG, 128);
            serverBootstrap.childOption(ChannelOption.SO_KEEPALIVE, true);
            // ChannelFuture描述的时异步回调的处理操作
            ChannelFuture future = serverBootstrap.bind(HostInfo.PORT).sync();
            future.channel().closeFuture().sync();// 等待Socket被关闭
        } finally {
            workerGroup.shutdownGracefully() ;
            bossGroup.shutdownGracefully() ;
        }
    }
}

public class HttpServerHandler extends ChannelInboundHandlerAdapter {
    private HttpRequest request;
    private DefaultFullHttpResponse response ;
    private HttpSession session ;
    private ChannelHandlerContext ctx ;

    /**
     * 依据传入的标记内容进行是否向客户端Cookie中保存有SessionId数据的操作
     * @param exists
     */
    private void setSessionId(boolean exists) {
        if(exists == false) {    // 用户发送来的头信息里面不包含有SessionId内容
            String encodeCookie = ServerCookieEncoder.STRICT.encode(HttpSession.SESSIONID, HttpSessionManager.createSession()) ;
            this.response.headers().set(HttpHeaderNames.SET_COOKIE,encodeCookie) ;// 客户端保存Cookie数据
        }
    }

    /**
     * 当前所发送的请求里面是否存在有指定的 SessionID数据信息
     * @return 如果存在返回true，否则返回false
     */
    public boolean isHasSessionId() {
        String cookieStr = this.request.headers().get(HttpHeaderNames.COOKIE) ; // 获取客户端头信息发送来的Cookie数据
        if (cookieStr == null || "".equals(cookieStr)) {
            return false ;
        }
        Set<Cookie> cookieSet = ServerCookieDecoder.STRICT.decode(cookieStr);
        Iterator<Cookie> iter = cookieSet.iterator() ;
        while(iter.hasNext()) {
            Cookie cookie = iter.next() ;
            if(HttpSession.SESSIONID.equals(cookie.name())) {
                if (HttpSessionManager.isExists(cookie.value())) {
                    this.session = HttpSessionManager.getSession(cookie.value()) ;
                    return true ;
                }
            }
        }
        return false ;
    }


    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
        this.ctx = ctx ;
        if (msg instanceof HttpRequest) {    // 实现HTTP请求处理操作
            this.request = (HttpRequest) msg; // 获取Request对象
            System.out.println("【Netty-HTTP服务器端】uri = " + this.request.uri() + "、Method = " + this.request.method() + "、Headers = " + request.headers());
            this.handleUrl(this.request.uri());
        }
    }

    private void responseWrite(String content) {
        ByteBuf buf = Unpooled.copiedBuffer(content,CharsetUtil.UTF_8) ;
        this.response = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1,HttpResponseStatus.OK,buf) ;
        this.response.headers().set(HttpHeaderNames.CONTENT_TYPE,"text/html;charset=UTF-8") ; // 设置MIME类型
        this.response.headers().set(HttpHeaderNames.CONTENT_LENGTH,String.valueOf(buf.readableBytes())) ; // 设置回应数据长度
        this.setSessionId(this.isHasSessionId());
        ctx.writeAndFlush(this.response).addListener(ChannelFutureListener.CLOSE) ; // 数据回应完毕之后进行操作关闭
    }



    private void sendImage(String fileName) throws Exception {
        String filePath = DiskFileUpload.baseDirectory + fileName ;
        File sendFile = new File(filePath) ;
        HttpResponse imageResponse = new DefaultHttpResponse(HttpVersion.HTTP_1_1,HttpResponseStatus.OK) ;
//        imageResponse.headers().set(HttpHeaderNames.CONTENT_LENGTH,String.valueOf(sendFile.length())) ;
        MimetypesFileTypeMap mimeMap = new MimetypesFileTypeMap() ;
        imageResponse.headers().set(HttpHeaderNames.CONTENT_TYPE,mimeMap.getContentType(sendFile)) ;
        imageResponse.headers().set(HttpHeaderNames.CONNECTION,HttpHeaderValues.KEEP_ALIVE) ;
        this.ctx.writeAndFlush(imageResponse) ;
        this.ctx.writeAndFlush(new ChunkedFile(sendFile)) ;
        // 在多媒体信息发送完毕只后需要设置一个空的消息体，否则内容无法显示
        ChannelFuture channelFuture = this.ctx.writeAndFlush(LastHttpContent.EMPTY_LAST_CONTENT) ;
        channelFuture.addListener(ChannelFutureListener.CLOSE) ;
    }

    public void handleUrl(String uri) {
        if ("/info".equals(uri)) {
            this.info();
        } else if ("/favicon.ico".equals(uri)) {
            this.favicon();
        } else if ("/show.png".equals(uri)) {
            this.show() ;
        }
     }
    public void info() {
        String content =
                "<html>" +
                        "  <head>" +
                        "       <title>Hello Netty</title>" +
                        "   </head>" +
                        "   <body>" +
                        "       <h1>好好学习，天天向上</h1>" +
                        "       <img src='/show.png'>" +
                        "   </body>" +
                        "</html>";   // HTTP服务器可以回应的数据就是HTML代码
        this.responseWrite(content);
    }

    public void favicon() {
        try {
            this.sendImage("favicon.ico");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
     public void show() {
         try {
             this.sendImage("show.png");
         } catch (Exception e) {
             e.printStackTrace();
         }
     }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) throws Exception {
        cause.printStackTrace();
        ctx.close();
    }
}

```




