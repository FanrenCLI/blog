---
title: IO
date: 2021-12-15 23:47:54
categories:
  - JAVA
tags:
  - IO
author: Fanrencli
---
	
## IO操作

### IO核心类
- `File`
- `InputStream`
- `OutputStream`
- `Reader`
- `Writer`
- `Serializable`

### File操作文件
- 构造方法：`File(String filepath)`
- 创建文件：`createNewFile()`
- 删除文件：`delete()`
- 路径符号：`File.separator`
- 获取父路径：`getParentFile()`
- 创建目录：`mkdir()` 和`mkdirs()`创建一级目录和多级目录
- 获取文件大小：`long lenght()`
- 判断是否是文件或者路径：`boolean isFile()` `boolean isDirectory()`
- 获取最近的修改时间：`long lastModified()`
- 获取文件名称：`getName()`
- 输出路径中包含的信息(可能是文件也可能是路径)：`String[] list()` `File[] listFiles()`


```java
public class Main {
    public static void main(String[] args) throws IOException {
        File file = new File("test.txt");
        System.out.println(file.createNewFile());
        System.out.println(file.delete());
    }
}
```

### 字节流与字符流
- 通过File定义一个需要操作的文件
- 通过字节流或者字符流的子类对象为父类对象进行实例化
- 进行数据的读写操作
- 关闭资源

#### 字节流

1. `InputStream` 抽象类

	- 读取单个字节并返回数据：`int read() throws IOException`
	- 读取数据保存在字节数组：`int read(byte[] b) throws IOException`
	- 读取数据保存在数组某个部分：`int read(byte[] b, int off, int len) throws IOException`
	- 返回值为int型，如果读取完毕则返回-1，否则返回读取的长度。
	- 子类：`FileInputStream`：
    	- 构造方法：`FileInputStream(File file)`

```java
public class Main {
    public static void main(String[] args) throws IOException {
        File file = new File("test.txt");
        InputStream inputStream = new FileInputStream(file);
        byte data[] = new byte[1024];
        int len = inputStream.read(data);
        inputStream.close();
        System.out.println(new String(data,0,len));
    }
}
```

```java
public class Main {
    public static void main(String[] args) throws IOException {
        File file = new File("test.txt");
        InputStream inputStream = new FileInputStream(file);
        byte data[] = new byte[1024];
        int foot=0;
        int temp=0;
        while((temp = inputStream.read())!=-1){
            data[foot++] = (byte)temp;
        }
        inputStream.close();
        System.out.println(new String(data,0,foot));

    }

}
```
2. `OutputStream`抽象类

	- 输出单个字节：`void write(int b) throws IOException`
	- 输出全部字节数组：`void write(byte[] b) throws IOException`
	- 输出部分字节数组：`void write(byte[] b, int off, int len) throws IOException`
	- 子类：`FileOutputStream`：
    	- 构造方法：`FileOutputStream(File file)`覆盖创建写入/`FileOutputStream(File file, boolean append)`是否追加写入

```java
public class Main {
    public static void main(String[] args) throws IOException {
        File file = new File("test.txt");
        OutputStream outputStream = new FileOutputStream(file);
        String str = "好好学习！！！";
        byte data[] = str.getBytes();
        outputStream.write(data);
        outputStream.close();

    }

}
```

#### 字符流

1. `Reader`
    - 输出全部字符数组：`int read(char[] b) throws IOException`
	- 读取数据保存在数组某个部分：`int read(char[] b, int off, int len) throws IOException`
	- 返回值为int型，如果读取完毕则返回-1，否则返回读取的长度。
    - 子类：`FileReader`
        - 构造方法：`FileReader(File file)`


```java
public class Main {
    public static void main(String[] args) throws IOException {
        File file = new File("test.txt");
        Reader reader = new FileReader(file);
        char data[] = new char[1024];
        int len = reader.read(data);
        System.out.println(new String (data,0,len));
        reader.close();
    }
}
```

2. `Writer`

    - 输出全部字符数组：`void write(char[] b) throws IOException`
    - 输出字符串：`void write(String str) throws IOException`
    - 子类：`FileWriter`
        - 构造方法：`FileWriter(File file)`覆盖创建写入/`FileWriter(File file, boolean append)`是否追加写入


```java
public class Main {
    public static void main(String[] args) throws IOException {
        File file = new File("test.txt");
        Writer writer = new FileWriter(file);
        String str = "我是谁？";
        writer.write(str);
        writer.close();
    }
}
```

#### 字节流与字符流的转换

字节流与字符流的区别在于：字节流直接与终端交互，而字符流需要通过缓冲区处理后进行输出。由于缓冲区操作的区别，从而导致如果字符流的输入输出不对资源进行关闭操作或者`flush()`方法，最终的文件不会出现对应的内容。

关于字节流与字符流：
- 如果有中文操作则优先选择字符流，否则优先选择字节流操作，因为后期所有关于网络通信的操作都会设计到字节的处理。
- 字节流与字符流的转换主要包含两个类：`InputSreamReader`&`OutputStreamWriter`对应与`Reader` 和 `Writer`两个类的子类。
- 构造方法：
    - `InputSreamReader(InputSream in)` 和 `OutputStreamWriter(OutputStream out)`通过这两个构造函数接受对应的输入输出的字节流转换到对应的字符流。


```java
public class Main {
    public static void main(String[] args) throws IOException {
        File file = new File("test.txt");
        OutputStream outputStream= new FileOutputStream(file);
        Writer writer = new OutputStreamWriter(outputStream);
        writer.write("hello world");
        writer.close();
    }
}
```

### 总结
- 字节流处理是计算机的主流处理方式，因为在内存中数据的存储都是按照字节的方式进行存储的，而由于中文的存在需要转换为字符流进行操作，但是其原来的内容都是以字节型数据进行展示
- 由于数据原来都是按照字节存储，所以对应的 `FileReader`&`FileWriter`是对应的 `InputSreamReader`&`OutputStreamWriter`的子类，其内部原理就是通过读取字节流数据，然后通过转换操作将字节数据转换为字符数据。

### 综合实践：实现文件复制操作

```java
public class Main {
    public static void main(String[] args) throws IOException {
        File file = new File("test.txt");
        File file1 =new File("test1.txt");
        InputStream inputStream = new FileInputStream(file);
        OutputStream outputStream= new FileOutputStream(file1);
        byte data[] = new byte[1024];
        int temp = 0;
        while((temp = inputStream.read(data))!=-1){
            outputStream.write(data,0,temp);
        }
        inputStream.close();
        outputStream.close();
    }
}
```

#### ByteArrayOutputStream和ByteArrayInputStream 

- 两个类作为内存流操作类，主要由于磁盘读写速度低于内存读写，所以通过用这个类作为某些操作的方法
- `ByteArrayOutputStream`主要用于将多段数据来源合并在一起进行`流`输出，因为网络传输数据一般都是分段传输的
- `ByteArrayInputStream` 主要用于将一整段数据以多段的方式以`流`的方式进行输出。
- 在java中数据交互很多时候都是以流的方式进行，所以用这两个类优于用字节数组操作。
- 而且资源不需要关闭

```java
public static void main(String[] args) throws IOException {
		String str = "123456789";//数据源
		ByteArrayInputStream in = new ByteArrayInputStream(str.getBytes());
		int read = in.read();//从这个输入流中读取下一个字节 返回一个无符号 的byte值，范围 0-255
		System.out.println((char)read);//输出结果为 "1"
		byte[] b = new byte[4];
		in.read(b);
		System.out.println(new String(b));//输出结果为 "2345"
		in.read(b, 0, 4);
		System.out.println(new String(b));//输出结果为 "6789"
	}
public static void main(String[] args) throws IOException {
    ByteArrayOutputStream os = new ByteArrayOutputStream();
    os.write(100);//将指定的字节写入此字节数组输出流。ps:虽然参数是int类型 但是只会写入8位，即一个字节
    os.write(new byte[] {0,0,0,100});//将字节数组写入内存
    os.write(new byte[] {0,0,0,100}, 0, 4);//将字节数组指定位置的数据写入内存
    byte[] byteArray = os.toByteArray();//获取写入内存流中的所有数据
    System.out.println(byteArray.length);//输入结果为9
}
```

#### 打印流(数据输出)

- 在输入输出流中，所有的数据都需要转换为String类然后再转为字节数据类型进行输入输出操作，这在我们日常开发中是非常不方便的。当我们希望将基本数据类型直接输出到文件中时，总是需要自己手动转换格式进行输出，因此打印流就此出现
- 包含有`PrintStream`和`PrintWriter`,分别继承于`OutputStream`和`Writer`

```java

public class Main {
    public static void main(String[] args) throws IOException {
        File file = new File("test.txt");
        PrintStream printStream = new PrintStream(new FileOutputStream(file));
        printStream.print(123123123);
        printStream.close();
    }

}
```

#### 扫描流（数据输入）

- 在打印流中主要解决了数据输出所遇到的问题，而在数据输入中主要使用`Scanner`进行解决。
- 构造函数：`public Scanner(InputStream source)`

```java
public class Main {
    public static void main(String[] args) throws IOException {
        Scanner scanner = new Scanner(System.in);
        System.out.print("请输入内容：");
        while(scanner.hasNextDouble()){
            System.out.println(scanner.nextDouble());
        }
    }
}
public class Main {
    public static void main(String[] args) throws IOException {
        Scanner scanner = new Scanner(System.in);
        System.out.print("请输入内容：");
        while(scanner.hasNext("\\d{4}")){
            System.out.println(scanner.next("\\d{4}"));
        }
    }
}
public class Main {
    public static void main(String[] args) throws IOException {
        Scanner scanner = new Scanner(new FileInputStream(new File("test.txt")));
        System.out.print("请输入内容：");
        scanner.useDelimiter("\\n");
        while(scanner.hasNext()){
            System.out.println(scanner.next());
        }
    }
}
```