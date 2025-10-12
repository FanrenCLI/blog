---
title: Influxdb
date: 2025-09-20 12:58:00
cover: true
top: true
categories:
  - SQL
tags:
  - Java
author: Fanrencli
---

## Influxdb

### Influxdb简介

Influxdb是一个开源的时序数据库，由Go语言编写而成，由InfluxData公司开发和维护。它被设计用于处理大量时间序列数据，例如监控数据、物联网数据、金融数据等。Influxdb具有以下特点：

- 高性能：Influxdb使用Go语言编写，具有高性能和低延迟的特点。
- 灵活的数据模型：Influxdb使用一种称为TSM（Time Series Merge）的数据模型，可以高效地存储和查询时间序列数据。
- 相比于Prometheus，Influxdb更适合存储和查询大量的时间序列数据，而Prometheus更适合存储和查询少量的时间序列数据。
- 数据只写不改：Influxdb的数据只写不改，一般之用来存储表达某种状态的数据 ，比如温度、内存使用率、磁盘使用率等。