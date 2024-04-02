---
title: GoLang
date: 2023-06-22 14:11:00
categories:
  - Go
tags:
  - Go knowledge
author: Fanrencli
---

## 家庭收支账簿
```go
package util
import (
	"fmt"
	"strconv"
)

type FamilyAccount struct{
	key string

	loop bool

	balance float64

	money float64

	note string

	detail string
	flag bool

}
func (this *FamilyAccount) income(){
	fmt.Println("本次收入金额：")
	fmt.Scanln(&this.money)
	this.balance +=this.money
	fmt.Println("本次收入说明：")
	fmt.Scanln(&this.note)
	this.detail += "\n收入\t"+strconv.FormatFloat(this.balance,'f',-1,64)+"\t"+strconv.FormatFloat(this.money,'f',-1,64)+"\t"+this.note
	this.flag=true
}
func (this *FamilyAccount) outcome(){
	fmt.Println("本次支出金额：")
	fmt.Scanln(&this.money)
	if this.money>this.balance{
		fmt.Println("余额不足")
	}		
	this.balance-=this.money
	fmt.Println("本次支出说明：")	
	fmt.Scanln(&this.note)
	this.detail +=	"\n支出\t"+strconv.FormatFloat(this.balance,'f',-1,64)+"\t"+strconv.FormatFloat(this.money,'f',-1,64)+"\t"+this.note
	this.flag=true
}
func (this *FamilyAccount) loginout(){
	fmt.Println("真的要推出么？(y/n)")
	choice:=""
	for{
		fmt.Scanln(&choice)
		if choice=="y"{
			this.loop=false
			break
		}
		if choice=="n"{
			break
		}
		
		fmt.Println("真的要退出么？(y/n)")

	}
}
func (this *FamilyAccount) printdetail(){
	if !this.flag{
		fmt.Println("当前没有收支信息！！！")
		// break
	}
	fmt.Println("-----------------当前收支明细--------------------")
	fmt.Println(this.detail)
}
func (this *FamilyAccount) MainMenu(){
	for {
		fmt.Println("\n-----------------家庭收支账本--------------------")
		fmt.Println("                 1 收支明细")
		fmt.Println("                 2 登记收入")
		fmt.Println("                 3 登记支出")
		fmt.Println("                 4 退出软件")
		fmt.Print("请选择（1-4）：")
		fmt.Scanln(&this.key)
		switch this.key {
			case "1":
				this.printdetail()
			case "2":
				this.income()
			case "3":
				this.outcome()
			case "4":
				this.loginout()
			default:
				fmt.Println("请输入正确的选项。。。")
		}
		if !this.loop {
			break 
		}
	}
	fmt.Println("你退出了家庭收支账户")	
}

func NewFamilyAccount() *FamilyAccount{
	return &FamilyAccount{
		key:"",
		loop:true,
		balance :10000.0,
		money :0.0,
		note :"",
		detail :"收支\t账户金额\t收支金额\t说   明",
		flag:false,
	}
}
```

```go
package main

import "fmt"

type Writer interface {
    Write([]byte) (int, error)
}

type StringWriter struct {
    str string
}

func (sw *StringWriter) Write(data []byte) (int, error) {
    sw.str += string(data)
    return len(data), nil
}

func main() {
    var w Writer = &StringWriter{}
    sw := w.(*StringWriter)
    sw.str = "Hello, World"
    fmt.Println(sw.str)
}
```