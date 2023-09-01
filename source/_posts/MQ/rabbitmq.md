---
title: RabbitMQ
date: 2023-9-2 00:01:00
categories:
  - MQ
tags:
  - RabbitMQ
author: Fanrencli
---
## 消息队列RabbitMQ

```java
class RabbitMQUtils{
    public static Channel getChannel(){
        ConnectionFactory connectionFactory = new ConnectionFactory();
        connectionFactory.setHost("192.168.86.72");
        connectionFactory.setUsername("rabbitmq");
        connectionFactory.setPassword("rabbitmq");
        connectionFacotry.setPort(5672);
        Channel channel = connectionFactory.createChannel();
        return channel;
    }
}
public class producer{
    public static void main(String[] args) throws Execption{
        Channel channel = RabbitMQUtils.getChannel();
        // 开启消息发送确认
        channel.confirmSelect();
        // 开启异步确认消息，第一个参数为成功接收参数时的回调，第二个参数为发布失败时的回调
        channel.addConfirmListener((item1,item2)->{
            System.out.println(item1);
        },(item1,item2)->{
            System.out.println(item1);
        })
        /**
         * 1.队列名称
         * 2.是否持久化队列
         * 3.是否重复消费
         * 4.是否自动删除
         * 5.其他属性
         * 
         */
        channel.queueDeclare("helloworld",true,false,false,null);
        int batchsize =0;
        for (int i=0; i<100; i++){
            batchsize++;
            /**
             * 1.交换机名称
             * 2.队列名称
             * 3.持久化消息
             * 4.消息内容
             */
            channel.basicPublish("","helloworld",MessageProperties.PRESISTENT_TEXT_PLAIN,("message"+i).getBytes());
            if (batchsize>=50){
                // 同步确认消息，批量确认
                if(channel.waitForConfirms()){
                    System.out.println("发布成功");
                    batchsize=0;
                }
            }
        }
    }
}
public class Consumer{

    public static void main(String[] args){
        Channel channel = RabbitMQUtils.getChannel();
        //  数值1：开启非公平队列，数值0：公平队列，数值>1:开启预取值，不同机器的性能不同
        channel.basicQos(1);
        /**
         * 1.队列名称
         * 2.是否自动应答
         * 3.接收到消息的回调
         * 4.消费失败的回调
         */
        channel.basicComsume("helloworld",false,(item1,item2)->{
            System.out.println("消息接受成功：" + new String(item2.getBody()));
            Thread.sleep(1000);
            // 1.消息的主键，2.是否批量确认
            channel.basicAck(item2.getEnvelope().getDeliverTag(),false);
        },item->{
            System.out.println(item+"消费中断");
        });
    }
}

```