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

- 生产者发消息只需要知道发给哪个交换机，设置对应的routingKey以及发送消息内容,同时还可以设置一些参数如消息的过期时间，是否持久化等
- 消费者只需要知道从哪个队列中接受消息，提供消费消息的方法和是否自动应答
- 队列的创建和交换机的创建可以由第三方创建，不需要在生产者和消费者的代码中涉及


### 简单模式以及工作队列模式

- 简单模式：生产者->消息队列<-消费者
- 工作队列模式：生产者->消息队列<-[消费者1,消费者2...]
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
             * 2.routingKey(简单模式下，routingKey就是队列名称）
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

### 发布订阅模式

- 广播模式(fanout)
- 直接模式(direct)
- 主题模式(topic)

#### 广播模式（fanout）

- 只需要一个交换机，然后生产者发送消息给交换机，不需要指定routingKey,消费者自己创建随机的队列并绑定到交换机上也不需要指定routingKey，然后就可以接受到消息。

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
        // 1.交换机名称，2.模式类型
        channel.exchangeDeclare("exchange_name","fanout");
        for (int i=0; i<100; i++){
            
            /**
             * 1.只填写交换机名称，队列名称不需要填写因为是广播模式
             */
            channel.basicPublish("exchange_name","",null,("message"+i).getBytes());
        }
    }
}
public class Consumer{

    public static void main(String[] args){
        Channel channel = RabbitMQUtils.getChannel();
        // 产生一个随机的队列，返回队列名
        String queueName = channel.queueDeclare().getQueue();
        // 绑定交换机和队列，routingKey不需要
        channel.queueBind(queueName,"exchange_name","");
        /**
         * 1.队列名称
         * 2.是否自动应答
         * 3.接收到消息的回调
         * 4.消费失败的回调
         */
        channel.basicComsume(queueName,true,(item1,item2)->{
            System.out.println("消息接受成功：" + new String(item2.getBody()));
        },item->{
            System.out.println(item+"消费中断");
        });
    }
}
```

#### 直接模式(direct)

- 相比与广播模式，直接模式需要指定routingKey,生产者发送消息给交换机的时候需要指定routingKey，然后交换机根据routingKey发送给对应的消费者的队列。消费者还是可以创建随机的队列，然后将队列绑定到交换机上同时在绑定的时候指定对应的routingKey

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
        // 1.交换机名称，2.模式类型
        channel.exchangeDeclare("exchange_name","direct");
        for (int i=0; i<100; i++){
            
            /**
             * 1.只填写交换机名称，队列名称不需要填写因为是广播模式
             */
            channel.basicPublish("exchange_name","white",null,("white"+i).getBytes());
            channel.basicPublish("exchange_name","black",null,("black"+i).getBytes());
            channel.basicPublish("exchange_name","grey",null,("grey"+i).getBytes());
        }
    }
}
public class Consumer1{

    public static void main(String[] args){
        Channel channel = RabbitMQUtils.getChannel();
        // 产生一个随机的队列，返回队列名
        String queueName = channel.queueDeclare().getQueue();
        // 绑定交换机和队列，routingKey不需要
        channel.queueBind(queueName,"exchange_name","black");
        channel.queueBind(queueName,"exchange_name","grey");
        /**
         * 1.队列名称
         * 2.是否自动应答
         * 3.接收到消息的回调
         * 4.消费失败的回调
         */
        channel.basicComsume(queueName,true,(item1,item2)->{
            System.out.println("消息接受成功：" + new String(item2.getBody()));
        },item->{
            System.out.println(item+"消费中断");
        });
    }
}
public class Consumer2{

    public static void main(String[] args){
        Channel channel = RabbitMQUtils.getChannel();
        // 产生一个随机的队列，返回队列名
        String queueName = channel.queueDeclare().getQueue();
        // 绑定交换机和队列，routingKey不需要
        channel.queueBind(queueName,"exchange_name","white");
        /**
         * 1.队列名称
         * 2.是否自动应答
         * 3.接收到消息的回调
         * 4.消费失败的回调
         */
        channel.basicComsume(queueName,true,(item1,item2)->{
            System.out.println("消息接受成功：" + new String(item2.getBody()));
        },item->{
            System.out.println(item+"消费中断");
        });
    }
}
```

#### 主题模式(topic)

- 主题模式相比于直接模式，直接模式的routingKey是写死的，只有完全匹配才会接收消息，而主题模式的routingKey可以适用通配符，只要和通配符相匹配就可以接受到消息
- 消息主题的通配符有两种，*代表一个单词，#代表0个单词或多个单词，主题模式的routingKey可以由多个单词组成，但是每个单词必须要通过`.`号分隔开
- 主题模式下的队列的routingKey，如果没有`*`或者`#`则变为直接模式，如果只有一个`#`则变为广播模式

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
        // 1.交换机名称，2.模式类型
        channel.exchangeDeclare("exchange_name","topic");
        for (int i=0; i<100; i++){
            
            /**
             * 1.只填写交换机名称，队列名称不需要填写因为是广播模式
             */
            channel.basicPublish("exchange_name","white.white1",null,("white1"+i).getBytes());
            channel.basicPublish("exchange_name","white.white2",null,("white2"+i).getBytes());
            channel.basicPublish("exchange_name","white.white3",null,("white3"+i).getBytes());
        }
    }
}
public class Consumer1{

    public static void main(String[] args){
        Channel channel = RabbitMQUtils.getChannel();
        // 产生一个随机的队列，返回队列名
        String queueName = channel.queueDeclare().getQueue();
        // 绑定交换机和队列，routingKey不需要
        channel.queueBind(queueName,"exchange_name","white.#");
        /**
         * 1.队列名称
         * 2.是否自动应答
         * 3.接收到消息的回调
         * 4.消费失败的回调
         */
        channel.basicComsume(queueName,true,(item1,item2)->{
            System.out.println("消息接受成功：" + new String(item2.getBody()));
        },item->{
            System.out.println(item+"消费中断");
        });
    }
}
```

### 死信

- 死信队列绑定到正常的队列中，依赖于正常的队列，当正常队列出现消息过期，消息拒绝，队列已满，则消息会转发到死信交换机上
- 死信交换机可以绑定到多个正常的队列中，根据每个正常队列绑定死信交换机的不同路由键，将正常队列的死亡的消息转发到死信交换机上对应路由键的队列。

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
    public static void inithandler(){
        Channel channel = RabbitMQUtils.getChannel();
        //声明交换机 1.交换机名称，2.模式类型
        channel.exchangeDeclare("nromal_exchange","direct");
        channel.exchangeDeclare("dead_exchange","direct");
        // 声明队列
        /**
         * 1.队列名称
         * 2.是否持久化队列
         * 3.是否重复消费
         * 4.是否自动删除
         * 5.其他属性
         * 
         */
        Map<String,Object> map = new HashMap<>();
        // 死信队列名称
        map.put("x-dead-letter-exchange","dead_exchange");
        // 死信队列路由
        map.put("x-dead-letter-routing-key","siwang");
        // 队列最大长度
        map.put("x-max-length","6");
        // 消息过期时间
        map.put("x-message-ttl","10000");
        // 正常队列绑定死信队列
        channel.queueDeclare("normal_queue",false,false,false,map);
        channel.queueDeclare("dead_queue",false,false,false,null);
        // 交换机与队列绑定
        channel.queueBind("normal_queue","nromal_exchange","zhengchang");
        channel.queueBind("dead_queue","dead_exchange","siwang");
    }
    public static void main(String[] args) throws Execption{
        inithandler();
        Channel channel = RabbitMQUtils.getChannel();
        for (int i=0; i<10; i++){
            // 发送正常的消息
            channel.basicPublish("normal_queue","zhengchang",
            new AMQP.BasicProperties().builder().expiration("10000").build(),
            ("test"+i).getBytes());
        }
    }
}
public class Consumer1{

    public static void main(String[] args){
        Channel channel = RabbitMQUtils.getChannel();
        // 正常队列的消费者,不要自动确认
        channel.basicComsume("normal_queue",false,(item1,item2)->{
            if (new String(item2.getBody()).contains("test5")){
                // 1.拒绝的消费实体，2.是否重新入队
                channel.basicReject(item2.getEnvelope().getDeliverTag(),false);
                System.out.println("拒绝消息：" + new String(item2.getBody()));
            }else{
                System.out.println("正常消息接收成功：" + new String(item2.getBody()));
            }
        },item->{
            System.out.println(item+"消费中断");
        });

        Channel channel1 = RabbitMQUtils.getChannel();
        // 死信队列消费者
        channel1.basicComsume("dead_queue",true,(item1,item2)->{
            System.out.println("死信消息接受成功：" + new String(item2.getBody()));
        },item->{
            System.out.println(item+"消费中断");
        });
    }
}
```











