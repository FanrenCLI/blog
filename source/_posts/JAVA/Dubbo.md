---
title: Dubbo
date: 2023-11-24 20:43:44
categories:
  - RPC
tags:
  - DUBBO
author: Fanrencli
---

## DUBBO

### 简介
DUBBO简单来说就是一个RPC协议实现框架，一般与注册中心（Zookeeper）结合使用来提供服务发现和服务注册,当然也可以不和注册中心结合，直接使用直连的方式进行服务调用。除了提供RPC远程接口调用功能之外，DUBBO还可以实现接口的负载均衡，服务降级，容错等相关功能。此外还可以通过额外扩展DUBBO的监控界面对当前的相关接口进行管理。

### 使用

使用DUBBO主要涉及以下几个步骤：
1. 引入依赖

```xml
<!-- dubbo核心依赖 -->
<dependency>
	<groupId>org.apache.dubbo</groupId>
	<artifactId>dubbo-spring-boot-starter</artifactId>
	<version>3.0.7</version>
</dependency>
<!-- 使用哪个通信协议 -->
<dependency>
	<groupId>org.apache.dubbo</groupId>
	<artifactId>dubbo-rpc-dubbo</artifactId>
	<version>3.0.7</version>
</dependency>
<!-- 使用哪个注册中心 -->
<dependency>
	<groupId>org.apache.dubbo</groupId>
	<artifactId>dubbo-registry-zookeeper</artifactId>
	<version>3.0.7</version>
</dependency>
```

2. 配置相关信息

```yml
# dubbo应用的名称，如果没有则默认使用app.name
dubbo.application.name = mydubbo
# 使用哪个协议
dubbo.protocol.name = dubbo
# dubbo通信的端口
dubbo.protocol.port = 20800
# 注册中心的地址
dubbo.registry.address = zookeeper://localhost:2181
```

```java
// 定义接口模板
public interface UserService{
	public String getUser(String name);
}
// 定义接口实现类,可存在多个实现类，通过版本区分
@DubboService(version="1.0")
public class UserServiceImp implements UserService{
	public String getUser(String name){
		return name;
	}
}
@DubboService(version="2.0")
public class UserServiceImp2 implements UserService{
	public String getUser(String name){
		return "lujie2";
	}
}

// 定义服务消费方,通过Spring注解进行管理
@Service
public class OrderService{
	// 通过dubbo3.0提供的注解进行服务注入，在服务启动时，通过扫描此注解，来生成代理类访问服务提供者
	@DubboReference(version='1.0')
	private UserService userService;
	public void getOrder(){
		return userService.getUser("lujie");
	}

}
```

### 原理
Dubbo底层原理实际上是一个Netty的服务器，监听的20880端口。服务消费方，只需要知道接口，然后通过代理的方式生成对应的实现类，这个实现类方法就是接收一个入参，然后通过网络将这个入参发送给服务提供方（多个服务提供方如何选择？负载均衡、限流。如果调用失败怎么办？容错机制），然后提供方在本地运行要调用的方法后将结果再通过网络返回给调用方。而zk的作用就是让消费者发现提供者的服务地址。因为需要网络传输数据，因此dubbo支持很多通信协议。此外，消费者一旦通过服务提供方发现了服务，会缓存到本地，因此如果zk崩溃了，也可以实现服务的调用。

#### dubbo2.x与dubbo3.x的不同
在2.x版本中，服务的提供方式是以接口为粒度的，一个接口对应zk上的一个节点，这样就导致一个接口有不同的版本时对内存的消耗较多，而3.x版本对这个情况进行了改善，在兼容2.x的版本之上，将服务提供的粒度转为以应用为粒度，同时在应用配置中我们可以通过参数对这种模式进行控制

```yml
dubbo.application.registry-mode=instance
dubbo.application.registry-mode=all
dubbo.application.registry-mode=interface
```

#### Triple

dubbo3.x版本提供了一个全新的RPC协议：Triple，基于HTTP2且完全兼容gRPC,这使得java服务也可以通过dubbo调用其他语言的服务：go

```xml
<!-- 引入triple协议依赖 -->
<dependency>
	<groupId>org.apache.dubbo</groupId>
	<artifactId>dubbo-rpc-triple</artifactId>
	<version>3.0.7</version>
</dependency>
```

```yml
dubbo.application.name=triple
```

#### 使用Triple协议的新特性

```xml
<dependency>
	<groupId>org.apache.dubbo</groupId>
	<artifactId>dubbo-common</artifactId>
	<version>3.0.7</version>
</dependency>
```

```java
public interface UserService{

	// 定义一个方法，这个方法，入参需要包含StreamObServer<String>参数
	public String getString(String name, StreamObServer<String> response);
}
// 通过注解声明一个dubbo服务
@DubboService(version="1.0")
public class UserServiceImp implements UserService{
	public String getString(String name, StreamObServer<String> response){
		response.onNext("have next":name);
		response.onNext("have next":name);
		response.onCompleted();
		return "success";
	}
}

// 定义服务消费方,通过Spring注解进行管理
@Service
public class OrderService{
	// 通过dubbo3.0提供的注解进行服务注入，在服务启动时，通过扫描此注解，来生成代理类访问服务提供者
	@DubboReference(version='1.0')
	private UserService userService;
	public void getOrder(){
		String res = userService.getString("lujie", new StreamObServer<String>(){
			@Override
			public void onNext(String data){
				System.out.println(data);
			}
		})

		return res;
	}

}
```

- 上述的只是服务端流的使用方法，这个只是单向的，还有一种双向的


```java
public interface UserService{

	// 定义一个方法，这个方法，入参需要包含StreamObServer<String>参数
	public StreamObServer<String> response getString(String name, StreamObServer<String> response);
}
// 通过注解声明一个dubbo服务
@DubboService(version="1.0")
public class UserServiceImp implements UserService{
	public StreamObServer<String> getString(String name, StreamObServer<String> response){
		return new StreamObServer<String>(){
			@Override
			public void onNext(String data){
				System.out.println("接收到调用方发来的信息"+data+",开始处理。。。");
				response.onNext("have next":name);
				response.onNext("have next":name);
				response.onCompleted();
			}
		};
	}
}

// 定义服务消费方,通过Spring注解进行管理
@Service
public class OrderService{
	// 通过dubbo3.0提供的注解进行服务注入，在服务启动时，通过扫描此注解，来生成代理类访问服务提供者
	@DubboReference(version='1.0')
	private UserService userService;
	public void getOrder(){
		StreamObServer<String> res = userService.getString("lujie", new StreamObServer<String>(){
			@Override
			public void onNext(String data){
				System.out.println("接收到提供方发来的信息"+data+",开始处理。。。");
			}
		});
		res.onNext("have next":name);
		res.onNext("have next":name);
		res.onCompleted();
	}

}
```

#### 跨语言RPC调用
java服务可以通过go语言进行调用，原理主要是通过protobuf来实现，protobuf作为桥梁，首先通过protobuf定义接口，然后利用protobuf编译器来编译定义的接口文件（xxx.proto），然后可以生成java对应的接口，然后用java语言实现这个接口的方法，并启动服务。在go语言中，复制一份java编译的xxx.proto文件，通过编译器生成go语言的相关代码，然后通过go语言来访问即可。

#### SPI

SPI是java远程提供的一个动态加载接口实现类的机制，如果在一个项目中定义个一个接口User，但是根据引入的不同的子项目想要不同的实现，此时就可以采用SPI机制。在子项目中定义User的实现类，此时只需要在jar包中的这个路径下创建META-INF/services/com.example.MyService文件，文件内容为`com.example.MyServiceImpl`，我们想要的实现方式。在主项目中引入这个子项目后，通过`java.util.ServiceLoader`就可以加载到这个实现类。如果同时引入了多个子项目就可以加载多个实现类，类似于过滤器Filter就可以通过这种方式进行扩展。

```java

public interface Animal {
 void run();
}
public class Cat implements Animal{
 @Override
 public void run() {
      System.out.println("cat run");
   }
}

public class Dog implements Animal {
 @Override
 public void run() {
      System.out.println("dog run");
   }
}
// 每次全量加载，一般JDBC就通过这种方式实现，提供接口，不同的jdbc实现类架加载
public static void main(String[] s){
	System.out.println("======this is SPI======");
	ServiceLoader serviceLoader = ServiceLoader.load(Animal.class);
	Iterator animals = serviceLoader.iterator();
	while (animals.hasNext()) {
		animals.next().run();
	}
}
```

在$MATE-INF/services$下创建$org.example.spi.Animal$文件
```org.example.spi.Animal
org.example.spi.Dog
org.example.spi.Cat
```



- dubbo中的SPI：在dubbo中类似于jdk原生提供的spi机制，自己实现了一套SPI，以下给出案例，dubbo中Filter实现的方式也是相同的，但是dubbo时按需加载，jdk时全部加载

在主项目中定义以下的接口：

```java
// 假设这是一个Dubbo服务接口,默认的实现类是MyService2，如果在子项目中有实现类则可以选择使用使用子项目实现类
@SPI("MyService")
public interface MyService {
    @Override
    public Result invoke(Invoker invoker, Invocation invocation);
}
```

在子项目中定义了两个实现类：

```java
public class MyService2 implements MyService {

    @Override
    public Result invoke(Invoker invoker, Invocation invocation) throws RpcException {
        // 在这里添加你自定义的过滤逻辑，比如打印日志、校验权限等
        // ...
        
        // 调用下一个Filter或者执行实际服务调用
        Result result = invoker.invoke(invocation);

        // 对返回结果进行处理或后置操作
        // ...

        return result;
    }
}
public class MyService3 implements MyService {

    @Override
    public Result invoke(Invoker invoker, Invocation invocation) throws RpcException {
        // 在这里添加你自定义的过滤逻辑，比如打印日志、校验权限等
        // ...
        
        // 调用下一个Filter或者执行实际服务调用
        Result result = invoker.invoke(invocation);

        // 对返回结果进行处理或后置操作
        // ...

        return result;
    }
}
```

在项目的资源目录（如src/main/resourcesMETA-INF/dubbo/com.example.MyService）下创建文件

```txt
MyService=com.example.impl.MyService2
MyService3=com.example.impl.MyService3
```

将子项目引入到主项目中，启动应用时根据@Activate注解的属性按条件加载多个实现类，如果没有注解，需要代码实现选择哪个类，
```java
// 这个类加载器加载类的时候会读取@SPI的注解内容
ExtensionLoader loader = ExtensionLoader.getExtensionLoader(MyService.class);
MyService service = loader.getDefaultExtension(); // 获取@SPI默认实现
MyService service = loader.getActivateExtension(); // 获取@Activate注解类实现
// 或者
MyService customService = loader.getExtension("MyService3"); // 根据名称获取特定实现
```

- 使用@Activate注解实现Filter

在实际开发中，常用的一种就是实现自定义的Filter,通过@Activate注解，启动时根据条件加载到对应的调用链中

```java
// 使用@Activate注解激活过滤器，可以设置order属性决定执行顺序，condition属性用于条件激活等,注意如果设置了condition=false，那么就需要额外的配置了
@Activate(order = 100,group = {Constants.CONSUMER, Constants.PROVIDER}, value = "protocol eq 'dubbo'" before="anotherFilter")
public class MyCustomFilter implements Filter {

    @Override
    public Result invoke(Invoker invoker, Invocation invocation) throws RpcException {
        // 在RPC调用前后添加你的自定义逻辑
        System.out.println("MyCustomFilter is being executed before the RPC call.");
        
        Result result = invoker.invoke(invocation);

        System.out.println("MyCustomFilter is being executed after the RPC call.");

        return result;
    }
}
```
在项目的资源目录（如src/main/resources/META-INF/dubbo/com.alibaba.dubbo.rpc.Filter）下创建文件
```txt
myCustomFilter=your.package.name.MyCustomFilter