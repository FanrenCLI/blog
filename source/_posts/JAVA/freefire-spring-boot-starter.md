---
title: FreeFire
date: 2024-01-13 14:11:00
categories:
  - JAVA
tags:
  - Spring
author: Fanrencli
---
## FreeFire-Spring-boot-starter
### 项目背景
在任何一个产品中，产品质量是一个重要的评价标准。而单元测试作为开发阶段可使用的手段，通过单元测试可以极大提高代码的质量。而单元测试的编写却会给开发带来极大的工作复旦。因此，本项目希望通过界面化的UI来配合开发自动生成单元测试代码。同时，为了不侵入原来的项目代码，采用spring-boot-starter项目对本项目进行构建，在原项目中只需要引入此项目的jar包，启动后即可访问对应的Ui界面。
### 项目创建
以下图片展示的就是项目的具体目录结构，并在目录树中介绍各个文件目录的作用。注意如果项目中引入了thymeleaf，则页面入口的文件必须存放在templates文件目录下，且同时会拦截所有@controller注解的请路径，并自动拼接后缀。
![项目目录](http://39.106.34.39:4567/B9EB38A3-7B54-4b80-BCB2-79E8794F9E32.png)
```txt
├─main
│  ├─java
│  │  └─com
│  │      └─example
│  │          └─freefire
│  │              ├─aspector #通过拦截器拦截所需要生成单元测试用例的接口方法
│  │              ├─configurator #配置类文件，通过配置类文件对本项目的Bean交给spring容器控制
│  │              ├─controller # 本项目是web项目
│  │              ├─model # 相关封装类
│  │              ├─Service # 业务相关的文件
│  │              │  └─impl
│  │              └─utils # 重点生成单测文件额工具类
│  └─resources
│      ├─META-INF # spring-boot-starter项目要求在此目录下必须存在spring.factories文件，用于将本文件中的相关配置交给其他项目引入
│      │  ├─resources # 如果没有引入任何前端页面的jar包（thymeleaf）,这个文件夹为默认的资源文件目录之一
│      │  └─dubbo # 根据SPI基础，自定义的dubbo的filter需要再这个路径下面进行注册。
│      ├─static  # 如果没有引入任何前端页面的jar包（thymeleaf）,这个文件夹为默认的资源文件目录之一
│      │  ├─css
│      │  ├─fonts
│      │  └─js
│      └─templates # 如果引入了thymeleaf，则此目录为默认的资源文件路径，页面的入口文件必须在这里。
└─test
    └─java
        └─com
            └─example
                └─freefire
```
### 代码记录

- dubbo实现SPI过滤器,拦截接口

```java
@Slf4j
@Activate(group="consumer")
public class t2comsumerFilter implements Filter{

    public Result invoke(final Invoker<?> invoker, final Invocation invocation) throws RpcException{
        Result res = invoker.invoke(invocation)
    }
}
@Slf4j
@Activate(group="provider")
public class t2providerFilter implements Filter{

    public Result invoke(final Invoker<?> invoker, final Invocation invocation) throws RpcException{
        Result res = invoker.invoke(invocation)
    }
}

```

- 切面请求拦截代码

```java
package com.example.freefire.aspector;
import com.alibaba.fastjson.JSON;
import com.example.freefire.model.ClassObject;
import com.example.freefire.utils.HandleStringUtils;
import com.example.freefire.utils.SystemConstants;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;
@Aspect
@Service
@ConfigurationProperties(prefix = "freefire.methodaspect")
public class MehtodAspect {
    private final static String POINTCUT = "(execution(* com.hundsun.hsfund.o45.basepub..*Impl.*(..))"+
            "|| execution(* com.hundsun.hsfund.am4.basepub..*Impl.*(..)))"+
            "|| execution(* com.hundsun.hsfund.trade.basepub..*Impl.*(..)))"+
            "&& !execution(* com.hundsun.hsfund.trade.basepub..*Impl.istradedate(..))"+
            "&& !execution(* com.hundsun.hsfund.trade.basepub.common.impl.CommonServiceImpl.*(..)))";
    // 拦截到请求后，执行方法如果报错，后续代码不会运行，并且spring只会拦截容器中存在的bean的方法。如果自己new一个是不行的
    @Around("execution(* com.example.freefire.Service.impl..*.*(..)))")
    public Object aroundMethod(ProceedingJoinPoint joinPoint) throws Throwable {
        String classFullName = joinPoint.getTarget().getClass().getName();
        String methodName = joinPoint.getSignature().getName();
        Object[] args = joinPoint.getArgs();
        boolean flag = CollectionUtils.isEmpty(Arrays.stream(args).filter(Objects::nonNull).collect(Collectors.toList()));
        if (!SystemConstants.ClassInfo.containsKey(methodName.toLowerCase())){
            ClassObject classObject = new ClassObject();
            classObject.setMethodName(methodName);
            classObject.setClassName(classFullName);
            SystemConstants.ClassInfo.put(methodName.toLowerCase(),classObject);
        }
        // 如果没有入参，默认为请求获取类名和方法名
        if (flag){
            Object response= joinPoint.proceed();
            HandleStringUtils.syncGenerateTestFile(classFullName,null, JSON.toJSONString(response),methodName);
            return response;
        }
        String request = args.length > 1 ? JSON.toJSONString(args):JSON.toJSONString(args[0]);
        // TODO 调用自动生成代码接口
        Object response = joinPoint.proceed();
        HandleStringUtils.syncGenerateTestFile(classFullName,request,JSON.toJSONString(response),methodName);
        return response;
    }
}
```

- 配置文件代码

```java
package com.example.freefire.configurator;
import org.apache.catalina.connector.Connector;
import org.springframework.boot.web.embedded.tomcat.TomcatServletWebServerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
// 通过此配置文件统一扫描需要注入容器的bean
@Configuration
@ComponentScan(basePackages = {
        "com.example.freefire"
})
public class FreeFireConfiguration {
    }
```

- spring.factories文件

```java
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
com.example.freefire.configurator.FreeFireConfiguration
```
