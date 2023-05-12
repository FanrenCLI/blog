---
title: Docker
date: 2021-12-26 16:17:11
categories:
  - JAVA
tags:
  - Docker
author: Fanrencli
---

## Docker相关操作

- 创建一个容器：`docker run -d -it --name java8 --restart=always --privileged=true -p 8023:22 -v /root/temp:/root/container jdk8.11`
- 查看容器：`docker ps`/`docker ps -a`
- 启动容器：`docker start ID `/`docker restart ID `
- 停止容器：`docker stop ID`/`docker kill ID`
- 删除容器：`docker rm -f ID`

- 查看镜像：`docker images`
- 删除镜像：`docker rmi ID`
- 删除全部镜像：`docker rmi $(docker images -q)`
- 下载一个镜像：`docker pull mysql`

- 安装命令：`yum -y install docker-io`
- 启动：`service docker start`
- 查看版本：`docker version`
- 搜索镜像：`docker search mysql`:INDEX（仓库地址）|NAME（仓库名称）|STARS（喜欢程度）|OFFICIAL（是否是官方）|UTOMATED（是否提供dockerfile）
- 镜像加速器：使用阿里云加速配置
- 查看容器启动日志：`docker logs -f --details -t ID/Name`


## Dockerfile文件


```shell
基础环境
From java:8
作者
MAINTAINER Fanrencli
操作
ADD eureka-server-0.0.1-SNAPSHOT.jar /data/app.jar
暴露端口
EXPOSE　8761
#容器启动胡执行的命令
ENTRYPOINT ["java","-jar","/data/app.jar"]
```

- 构建镜像：`docker build -t 仓库名/镜像名:版本号 .[dockerfile文件路径]`
- 修改镜像名称：`docker tag 仓库名/镜像名:版本号 新仓库名/镜像名:版本号`
- 登陆docker：`docker login`
- 推送到官方仓库中：`docker push 仓库名/镜像名:版本号`

## 容器操作

- 进入容器：`docker exec -it NAME /bin/bash`
- 使用root用户进入容器：`docker exec -it -u root java8 /bin/bash`
- 重新生成镜像：`docker commit `
- 文件挂载：`docker run -itd -v 宿主机路径：容器路径 -P nginx`
- 容器互联：docker容器默认新建一个容器都会分配一下ip地址，各个容器之间可以通过各自的ip进行连接。或者通过 `docker run --link container1:container2`进行连接（单向）



