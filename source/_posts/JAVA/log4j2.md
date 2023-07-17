---
title: 日志文件解析
date: 2023-7-16 16:50:00
categories:
  - JAVA
tags:
  - Log4j2
author: Fanrencli
---

### 日志框架log4j2

#### XML文件解析

- 同步输出形式
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Configuration status="debug" name="MyApp" packages="">
    <!--全局Filter-->
    <ThresholdFilter level="ALL"/>
    <!-- Appenders节点指定了日志输出的类型 -->
    <Appenders>
        <Console name="Console" target="SYSTEM_OUT">
            <PatternLayout pattern="%d{HH:mm:ss.SSS} [%t] %-5level %logger{36} - %msg%n" />
        </Console>
        <RollingFile name="RollingFile" fileName="logs/app.log"
                     filePattern="logs/app-%d{yyyy-MM-dd HH}.log">
            <!--Appender的Filter-->
            <ThresholdFilter level="info" onMatch="ACCEPT" onMismatch="DENY"/>
            <PatternLayout>
                <Pattern>%d %p %c{1.} [%t] %m%n</Pattern>
            </PatternLayout>
            <Policies>
                <SizeBasedTriggeringPolicy size="500MB"/>
            </Policies>
        </RollingFile>
    </Appenders>
    <!-- Loggers节点指定了不同包的日志以何种类型进行输出，并定义日志级别 -->
    <!-- 其中Root节点指定了默认的输出形式 -->
    <Loggers>
        <Logger name="com.meituan.Main" level="trace" additivity="false">
            <!--Logger的Filter-->
            <ThresholdFilter level="debug"/>
            <appender-ref ref="RollingFile"/>
        </Logger>
        <Root level="debug">
            <AppenderRef ref="Console"/>
        </Root>
    </Loggers>
</Configuration>
```

- 异步输出日志形式
- 实现异步输入日志有两种方式：
  1. 通过AsyncAppender实现（不推荐）
  2. 通过AsyncLogger实现（推荐）
  3. 以上两种可以混合使用


```xml
<?xml version="1.0" encoding="UTF-8"?>
<Configuration status="debug" name="MyApp" packages="">
    <!--全局Filter-->
    <ThresholdFilter level="ALL"/>
    <!-- Appenders节点指定了日志输出的类型 -->
    <Appenders>
        <Console name="Console" target="SYSTEM_OUT">
            <PatternLayout pattern="%d{HH:mm:ss.SSS} [%t] %-5level %logger{36} - %msg%n" />
        </Console>
        <RollingFile name="RollingFile" fileName="logs/app.log"
                     filePattern="logs/app-%d{yyyy-MM-dd HH}.log">
            <!--Appender的Filter-->
            <ThresholdFilter level="info" onMatch="ACCEPT" onMismatch="DENY"/>
            <PatternLayout>
                <Pattern>%d %p %c{1.} [%t] %m%n</Pattern>
            </PatternLayout>
            <Policies>
                <SizeBasedTriggeringPolicy size="500MB"/>
            </Policies>
        </RollingFile>
        <!-- AsyncAppender就是用Async套一层 -->
        <Async name= "myAsync">
            <appender-ref ref = "Console">
        </Async>
    </Appenders>
    <!-- Loggers节点指定了不同包的日志以何种类型进行输出，并定义日志级别 -->
    <!-- 其中Root节点指定了默认的输出形式 -->
    <!-- Additivity指定是否继承Root -->
    <Loggers>
        <Logger name="com.meituan.Main" level="trace" additivity="false">
            <!--Logger的Filter-->
            <ThresholdFilter level="debug"/>
            <appender-ref ref="RollingFile"/>
        </Logger>
        <!-- AsyncLogger标签指定了异步输出的类型 -->
        <AsyncLogger name="com.meituan.Main" level="trace" additivity="false">
            <appender-ref ref="RollingFile"/>
            <appender-ref ref="myAsync"/>
        </AsyncLogger>
        <Root level="debug">
        <!-- 使用哪个类型就以那种类型输出 -->
            <AppenderRef ref="myAsync"/>
            <AppenderRef ref="Console"/>
        </Root>
    </Loggers>
</Configuration>
```