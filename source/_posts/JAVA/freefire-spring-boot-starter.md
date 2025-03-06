---
title: 单元测试高效生成方案
date: 2024-01-13 14:11:00
categories:
  - JAVA
tags:
  - Spring
author: Fanrencli
---
## FreeFire-Spring-boot-starter

### 项目背景

在任何一个产品中，产品质量是一个重要的评价标准。而单元测试作为开发阶段可使用的手段，通过单元测试可以极大提高代码的质量。而单元测试的编写却会给开发带来极大的工作负担。因此，本项目希望通过界面化的UI来配合开发自动生成单元测试代码。同时，为了不侵入原来的项目代码，采用spring-boot-starter项目对本项目进行构建，在原项目中只需要引入此项目的jar包，启动后即可访问对应的Ui界面。

### 项目思路

在实际项目中，如果使用SpringBoot框架，那么对应服务接口的访问方法可以通过spring框架提供的统一功能进行拦截，获取接口出入参数。针对数据库交互，可以通过mybatis提供的拦截器进行拦截，获取数据库交互的参数。同时，为了防止方法嵌套，导致数据库交互参数获取不到，可以通过切面进行拦截，防止方法嵌套。

![项目结构](http://fanrencli.cn/fanrencli.cn/unittest.png)

### 代码实现思路

- spring拦截器拦截接口访问，并通过ThreadLocal存储参数,隔离线程操作

```java
@Aspect
@Component
@Service
@Slf4j
public class MehtodInvokeAspect {

    @Autowired(required = false)
    private UnitTestMethodIntercept unitTestMethodIntercept;

    /**
     * 获取方法的入参和出参
     */
    @Around("@within(org.springframework.stereotype.Service)")
    public Object aroundMethod(ProceedingJoinPoint joinPoint) throws Throwable {

    }

```

- Mybatis拦截器拦截数据库交互

```java
@Slf4j
@Component
@Intercepts({
        @Signature(type = Executor.class, method = "query", args = {MappedStatement.class, Object.class, RowBounds.class, ResultHandler.class}),
        @Signature(type = Executor.class, method = "query", args = {MappedStatement.class, Object.class, RowBounds.class, ResultHandler.class, CacheKey.class, BoundSql.class}),
        @Signature(type = Executor.class, method = "update", args = {MappedStatement.class, Object.class})
})
public class MybatisInterceptor implements Interceptor {
    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        if (null == MethodUnitTestConstants.isInterceptedMethod()) return invocation.proceed();
        InvokeEntity invokeEntity = handleInvokeEntity(invocation);
        Object proceed = invocation.proceed();
        invokeEntity.setResponseParamJsonString(JSON.toJSONString(proceed, SerializerFeature.IgnoreErrorGetter));
        log.info("方法调用信息：{}", JSON.toJSONString(invokeEntity, SerializerFeature.IgnoreErrorGetter));
        MethodUnitTestConstants.getThreadLocalList().add(invokeEntity);
        this.buildInvokeEntityTree(invokeEntity);
        return null;
    }
..
}
```

- openapi开放数据交互接口，提高项目可扩展性，外部项目引入后可通过实现接口，自定义单元测试代码拦截逻辑

```java
public interface UnitTestMethodIntercept {
    /**
     * 单元测试方法拦截
     */
    boolean isInterceptedMehod(String className, String methodName);
    /**
     * 单元测试代码生成路径
     */
    String getUnitTestCodeFilePath();
}
```

- 启动定时任务，生成单元测试用例文件

```java
@Slf4j
@Component
public class SchedualTask {
    @Autowired
    private IUnitTestFileService unitTestFileService;
    @Scheduled(fixedDelay = 3000)
    public void genFileTask() throws InterruptedException{
        Thread.sleep(3000);
        unitTestFileService.generateTestFile();
    }
}
```

- spring.factories文件

```java
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
com.example.freefire.configurator.FreeFireConfiguration
```
