---
title: Vue实录
date: 2022-10-25 14:11:00
categories:
  - 前端
tags:
  - Vue
author: Fanrencli
---

## axios

- 导入axios组件后使用方式并非通过Vue.use来注册全局组件，可以通过原型链来调用具体使用方式为：`Vue.prototype.$axios = axios`,这样再其他组件中就可以通过`this.$axios`进行使用

```js
    import Vue from 'vue'
    import axios from "axios";
    // axios无法使用use来注册，所以采用原型链来全局使用
    Vue.prototype.$axios = axios
    const requests = axios.create({
        baseURL:"/api",
        timeout:5000
    })
    requests({
        url:'/product/getBaseCategoryList',
        method:'get'
    })
    requests.interceptors.request.use((config)=>{
        return config;
    });

    requests.interceptors.response.use((res)=>{
        return res.data;
    },(err)=>{
        return Promise.reject(new Error('fail'))
    });
    export default requests;
```
```js

    //发送get
    axios({
        method:"GET",
        url:"http://localhost:3000/posts/1",
        params:{
            title:"axios学习",
            author:"Yehaocong"
        }
    }).then(response=>{
        console.log(response);
    }).catch(function (error) {
        console.log(error);
    });

    //发送post
    axios({
        method:"POST",
        url:"http://localhost:3000/posts",
        data:{
            title:"axios学习",
            author:"Yehaocong"
    }
    }).then(response=>{
        console.log(response);
    }).catch(function (error) {
        console.log(error);
    });


    //其他发送请求的api
    //发送get,使用get，第一个参数时url，第二个参数时config配置对象
    axios.get("http://localhost:3000/posts/1")
    .then(response=>{
        console.log(response);
    }).catch(function (error) {
        console.log(error);
    });

    //发送post
    //发送post请求，第一个参数时url，第二个参数时请求体，第三个参数时config配置对象
    axios.post("http://localhost:3000/posts",
    {title:"axios学习2",
        author:"Yehaocong2"})
        .then(response=>{
        console.log(response);
    }).catch(function (error) {
        console.log(error);
    });
```

## vuex

- vuex的使用首先需要安装vuex插件 `npm install vuex@3`
- vuex一般再src项目下新建store文件夹，用于统一管理vuex，并且可以再这个目录下进行模块开发
`
```js
// Home
    import { reqCategoryList } from "@/api"
    const state = {
        categoryList:[]
    }
    const getters = {}
    const mutations = {
        CATEGORYLIST(state, categoryList) {
            state.categoryList = categoryList
        }
    }
    const actions = {
        async categoryList({ commit }) {
            let result = await reqCategoryList();
            if (result.code == 200) {
                commit('CATEGORYLIST', result.data)
            }
        }
    }

    export default {
        state,
        getters,
        mutations,
        actions
    }
```

```js
// Search
    const state = {

    }
    const getters = {}
    const mutations = {}
    const actions = {}

    export default {
        state,
        getters,
        mutations,
        actions
    }
```

```js
    import Vue from "vue";
    import Vuex from "vuex";
    Vue.use(Vuex);

    import Home from "./Home"
    import Search from "./Search";
    export default new Vuex.Store({
        modules:{
            Home,
            Search
        }
    })
```

## router

### 安装
- Vue-router作为插件使用，需要通过npm安装 `npm install vue-router@3/4`
- Vue-router使用一般在src目录下新建router文件夹
- 使用方式：1.`<router-view>`;2.`<router-link>`/`this.$router.push[replace]`
```js
import vue from "vue"
import VueRouter from "vue-router"
vue.use(VueRouter)
import Home from "@/pages/Home"
import Search from "@/pages/Search"
import Login from "@/pages/Login"
import Register from "@/pages/Register"
export default new VueRouter({
    routes:[
        {
            path:"/home",
            component:Home,
            meta:{show:true}
        },{
            name:"search",
            path:"/search/:keyword",
            component:Search,
            meta:{show:true}
        },{
            path:"/login",
            component:Login,
            meta:{show:false}
        },{
            path:"/register",
            component:Register,
            meta:{show:false}
        },{
            path:"*",
            redirect:"/home"
        }
    ]
})
```

## Promise & Async/await

```js
// 被asyc注释的函数返回一个Promise对象
async function foo(){
	return 1;
}
//等价于

function foo(){
	return new Promise((resolve,reject)=>{
        resolve(1);
    })
}
//await表达式会暂停当前 async function的执行，等待Promise处理完成。 如果await的promise失败了，就会抛出异常，需要通过try–catch来捕获处理。
async function foo(){
	try{
        let num = await 1;
        return num;
    }catch(err){
        TODO;
    }
}
foo().then((num)=>{console.log(num)})
//等价于
function foo(){
	return new Promise((resolve,reject)=>{
        if (/* 异步操作成功 */){
            resolve(value);
        } else {
            reject(error);
        }
    }).then((num) => console.log(num)).catch((err)=>TODO);
}
// 总的来说promise与async、await作用大致相同，async、await的出现是用于优化promise链式调用
```


## 小工具nprogress

- 发送请求时，在页面的上方显示进度条

```js
    import axios from "axios";
    // 必须引入以下两行代码
    import nProgress from "nprogress";
    import "nprogress/nprogress.css";

    const requests = axios.create({
        baseURL:"/api",
        timeout:5000
    })

    requests.interceptors.request.use((config)=>{
      // 请求发送的时候开始
        nProgress.start();
        return config;
    });

    requests.interceptors.response.use((res)=>{
      // 请求返回的时候结束
        nProgress.done();
        return res.data;
    },(err)=>{
        return Promise.reject(new Error('fail'))
    });


    export default requests;
```