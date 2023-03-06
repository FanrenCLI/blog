---
title: Vue概述
date: 2022-10-25 14:11:00
categories:
  - 前端
tags:
  - Vue
author: Fanrencli
---

## Vue安装

1. Node.js环境安装
  - 安装node环境之后，npm也会直接安装，通过`node -v`命令查看node的版本，`npm -v`查看npm的版本。
  - 其次npm安装之后有默认的安装位置，可以通过在user目录下新建`.npmrc`文件来配置npm相关的配置信息。但是建议只配置npm的下载镜像，其他配置不要更改。一旦更改配置之后，安装Vue后可能无法通过脚手架创建项目，还需要手动配置环境变量
2. 安装Vue
  - Vue2:建议全局安装,`npm install -g vue@2`这个命令会安装vue2版本的最新版，
  - Vue3:通过`npm install -g vue@latest`（不过vue3版本一般不需要手动安装vue3,可以直接创建项目顺带就安装vue3）
3. 安装脚手架
  - 建议全局安装：`npm install -g @vue/cli[@x]`通过这个命令可以安装3.x版本以上的脚手架
  - `npm install -g vue-cli` 通过这个命令可以安装2.x版本的脚手架
  - 但是Vue3版本后的脚手架官方建议采用`vite`进行使用,不用手动安装vite,创建项目时自动安装
4. 创建项目
  - 安装vue-cli是3.x版本以上的，可以通过`vue create [project_name]`创建项目
  - 安装vue-cli是2.x版本的，可以通过`vue init [webpack] [project_name]`创建项目，其中webpack是项目模板，一般建议采用webpack,也可以使用其他方式
  - 若使用vue3开发，可以直接通过`npm create[init] vite@latest/vue@latest project`创建项目，无需安装vue和vite
5. vue-cli和vite
  - vite:可以在没有安装vue和vite情况下，通过`npm create[init] vite@latest/vue@latest project`直接创建项目，顺便安装vite/vue3
  - vue-cli:必须安装vue和vue-cli，然后通过`vue create [name]`创建项目
  - vue2运行项目是`npm run serve`,vue3运行项目是`npm run dev`
  - `npm install `安装项目的依赖

## Vue2项目结构

- node_modules:放置项目依赖的地方
- public:一般放置一些共用的静态资源，打包上线的时候，public文件夹里面资源原封不动打包到dist文件夹里面
- src：程序员源代码文件夹
    - assets文件夹：经常放置一些静态资源（图片），assets文件夹里面资源webpack会进行打包为一个模块（js文件夹里面）
    - components文件夹:一般放置非路由组件（或者项目共用的组件）
        - App.vue 唯一的根组件
        - main.js 入口文件【程序最先执行的文件】
        - babel.config.js:babel配置文件
        - package.json：看到项目描述、项目依赖、项目运行指令
        - README.md:项目说明文件

### 新版扩展文件

不同的版本自动生成的配置文件都有一些变化，其中常用的文件有如下几个：

1. 关闭eslint校验工具

创建vue.config.js文件：需要对外暴露

```js
module.exports = {
   lintOnSave:false,
}
```
2. src文件夹的别名的设置

因为项目大的时候src（源代码文件夹）：里面目录会很多，找文件不方便，设置src文件夹的别名的好处，找文件会方便一些
创建jsconfig.json文件

```json
{
    "compilerOptions": {
        "baseUrl": "./",
        "paths": {
            "@/*": [
                "src/*"
            ]
        }
    },
    "exclude": [
        "node_modules",
        "dist"
    ]
}
```

## 文件代码解析

1. index.html
index.html为最终的页面文件，所有内容都会在这里展示，其中`id=app`为Vue文件控制的标签，最后打包的内容都将在这个标签内出现
```xml
<!DOCTYPE html>
<html lang="">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <link rel="icon" href="<%= BASE_URL %>favicon.ico">
    <link rel="stylesheet" href="<%= BASE_URL %>reset.css">
    <title><%= htmlWebpackPlugin.options.title %></title>
  </head>
  <body>
    <noscript>
      <strong>We're sorry but <%= htmlWebpackPlugin.options.title %> doesn't work properly without JavaScript enabled. Please enable it to continue.</strong>
    </noscript>
    <div id="app"></div>
    <!-- built files will be auto injected -->
  </body>
</html>

```
2. main.js
main.js文件作为Vue工程的入口文件，Vue项目启动编译打包都将从这个文件出发，这个文件主要用来做需要全局属性的事情（l例如引入插件，并使用插件），以及最重要的挂载index.html文件中的标签作为页面入口
```js
    import Vue from 'vue'
    import App from './App.vue'
    import router from '@/router'
    import TypeNav from '@/components/TypeNav'
    import store from "@/store"
    Vue.component(TypeNav.name,TypeNav)
    Vue.config.productionTip = false

    const vm = new Vue({
        render: h => h(App),
        router,
        store
    }).$mount('#app')
```
3. app.vue
app.vue作为统一管理所有组件的总组件，其中主要用于注册其他组件，以及main.js会引用此组件
```xml
  <template>
    <div id="app">
      <Header></Header>
      <router-view></router-view>
      <Footer v-show="$route.meta.show"></Footer>
    </div>
  </template>

  <script>

    export default {
      name: 'App',
      components: {
        Header,
        Footer
      }
    }
  </script>

  <style>

  </style>
```
