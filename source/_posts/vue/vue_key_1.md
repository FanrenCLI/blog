---
title: Vue实录
date: 2022-10-25 14:11:00
categories:
  - 前端
tags:
  - Vue
author: Fanrencli
---

## 数据代理&数据劫持
- 所谓数据代理就是通过另一个对象来代理真实对象的数据，通过代理对象操作真实对象
- 数据劫持就是在数据代理的基础上，通过数据代理对真实的对象的所有属性进行监视，封装一层成为代理对象，此后对真实数据的修改全部被代理对象劫持，然后由代理对象根据逻辑进行真实数据的修改
- `Object.defineProperty` 用于设置对象属性，监听对象是否被改变或被读取
- Vue实现数据双向绑定的原理就是基于`Object.defineProperty`，通过这个方法对数据的每一个属性提供get/set方法，从而实现数据对数据的监听绑定

```js
//真实数据
let Sex1 = male; 
// 代理数据
let person = {
  age:'13',
  name:'jone'
}
Object.defineProperty(person,'sex',{
  value:'female',
  enumerable:true //可以被遍历读取
  writable:true //可以被修改
  configurable:true //可以被删除
  get:function getSex(){
    return Sex1;
  },
  set:function setSex(value){
    Sex1 = value
  }
})

```

## 计算属性
```xml
<input v-model='firstName' type='text'>
<input v-model='lastName' type='text'>
<div>{{fullName}}</div>
```
```js
vm.$computed('fullName',{
  get:function getfullName(){
    return this.firstName+'-'+this.lastName;
  },
  set:function setfullName(value){
    const arr = value.split('-');
    this.firstName = arr[0];
    this.lastName = arr[1];
  }
})
const vm = new Vue({
  el:"app",
  data(){
    return {
      firstName:"jack",
      lastName:"json"
    }
  },
  computed:{
    // 当计算属性不会修改，只会读取时可以：
    fullName(){
      return this.firstName+'-'+this.lastName;
    }
    fullName:{
      // 初次读取时调用，依赖的属性改变时被调用
      get:function getfullName(){
        return this.firstName+'-'+this.lastName;
      },
      set:function setfullName(value){
        const arr = value.split('-');
        this.firstName = arr[0];
        this.lastName = arr[1];
      }
    }
  }
})
```

## 监视属性

```xml
<div>{{fullName}}</div>
```
```js
const vm = new Vue({
  el:"app",
  data(){
    return {
      fullName:"jack-json"
    }
  },
  watch:{
    //简写形式
    fullName(newvalue,oldvalue){
      console.log("新的值是:"+ newvalue);
      console.log("旧的值是："+ oldvalue);
    }
    // 除了data中的属性，计算属性也可以监视
    fullName:{
      // 上来就执行
      immediate:true,
      // 属性内部还有对象时，对象属性发生改变可以监视
      deep:true,
      // 数据发生改变时调用
      handler(newvalue,oldvalue){
        console.log("新的值是:"+ newvalue);
        console.log("旧的值是："+ oldvalue);
      }
    }
  }
})
vm.$watch('fullName',function(newvalue,oldvalue){
    console.log("新的值是:"+ newvalue);
    console.log("旧的值是："+ oldvalue);
  }
})
vm.$watch('fullName',{
  // 上来就执行
  immediate:true,
  // 属性内部还有对象时，对象属性发生改变可以监视
  deep:true,
  // 数据发生改变时调用
  handler(newvalue,oldvalue){
    console.log("新的值是:"+ newvalue);
    console.log("旧的值是："+ oldvalue);
  }
})
```

## 计算属性与监视属性对比

- 计算属性可以完成的属性监视属性都可以做到，但是计算属性做不到监视属性可以做到的事情：监视属性可以异步操作（定时器）
- 计算属性需要返回值，监视属性不需要返回值，一旦异步操作，计算属性的返回值就无法获取了。
