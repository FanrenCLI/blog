---
title: Spring
date: 2021-12-23 20:43:44
categories:
  - JAVA
  - Spring
tags:
  - AOP
  - IOC
  - MVC
author: Fanrencli
---

## IOC

### 四种注入方式

- 构造器注入
- setter注入
- 静态工厂
- 实例工厂

### Bean

- singleton:单例，默认作用域。
- prototype:原型，每次创建一个新对象。
- request:请求，每次Http请求创建一个新对象，适用于WebApplicationContext环境下。
- session:会话，同一个会话共享一个实例，不同会话使用不用的实例。
- global-session:全局会话，所有会话共享一个实例。

<p style="text-indent:2em">
创建的Bean如果是prototype：对于原型Bean,每次创建一个新对象，也就是线程之间并不存在Bean共享，自然是不会有线程安全的问题。

创建的Bean如果是：singleton，所有线程都共享一个单例实例Bean,因此是存在资源的竞争。
如果单例Bean,是一个无状态Bean，也就是线程中的操作不会对Bean的成员执行查询以外的操作，那么这个单例Bean是线程安全的。比如Spring mvc 的 Controller、Service、Dao等，这些Bean大多是无状态的，只关注于方法本身。

对于有状态的bean，Spring官方提供的bean，一般提供了通过ThreadLocal去解决线程安全的方法，比如RequestContextHolder、TransactionSynchronizationManager、LocaleContextHolder等。
</p>

## AOP

- 通知（Advice）: AOP 框架中的增强处理。通知描述了切面何时执行以及如何执行增强处理。
- 连接点（join point）: 连接点表示应用执行过程中能够插入切面的一个点，这个点可以是方法的调用、异常的抛出。在 Spring AOP 中，连接点总是方法的调用。
- 切点（PointCut）: 可以插入增强处理的连接点。
- 切面（Aspect）: 切面是通知和切点的结合。
- 引入（Introduction）：引入允许我们向现有的类添加新的方法或者属性。
- 织入（Weaving）: 将增强处理添加到目标对象中，并创建一个被增强的对象，这个过程就是织入。

```java
@Aspect
public class TransactionDemo {
    @Pointcut(value="execution(* com.yangxin.core.service.*.*.*(..))")
    public void point(){
    }
    @Before(value="point()")
    public void before(){
        System.out.println("transaction begin");
    }
    @AfterReturning(value = "point()")
    public void after(){
        System.out.println("transaction commit");
    }
    @Around("point()")
    public void around(ProceedingJoinPoint joinPoint) throws Throwable{
        System.out.println("transaction begin");
        joinPoint.proceed();
        System.out.println("transaction commit");
    }
}
```
## MVC

```xml
<web-app>
    <display-name>Archetype Created Web Application</display-name>
    <servlet>
        <servlet-name>mymvc</servlet-name>
        <servlet-class>com.fanrencli.mvcframe.servlet.MyDispatcherServlet</servlet-class>
        <init-param>
            <param-name>contextconfiglocation</param-name>
            <param-value>application.properties</param-value>
        </init-param>
        <load-on-startup>1</load-on-startup>
    </servlet>

    <servlet-mapping>
        <servlet-name>mymvc</servlet-name>
        <url-pattern>/</url-pattern>
    </servlet-mapping>
</web-app>
```

```java
public class MyDispatcherServlet extends HttpServlet {
    private Map<String,Object>ioc = new HashMap<>();
    private Properties contextconfig = new Properties();
    private List<String> classNames = new ArrayList<>();
    private Map<String,Method> handlerMapping = new HashMap<>();
    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        this.doPost(req,resp);
    }

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        try {
            doDispatch(req,resp);
        }catch (Exception e){
            e.printStackTrace();
            resp.getWriter().write("500");
        }
    }
    private void doDispatch(HttpServletRequest req, HttpServletResponse resp) throws Exception{
        String url = req.getRequestURI();
        String contextpath = req.getContextPath();
        url = url.replaceAll(contextpath,"").replaceAll("/+","/");
        if(!this.handlerMapping.containsKey(url)){
            resp.getWriter().write("404 Not Found");
        }
        Method method = this.handlerMapping.get(url);
        Map<String,String[]> params = req.getParameterMap();
        String beanName = tolowerfirst(method.getDeclaringClass().getSimpleName());
        method.invoke(ioc.get(beanName),new Object[]{req,resp,params.get("name")[0]});
    }

    // 核心处理流程
    @Override
    public void init(ServletConfig config) throws ServletException {
        System.out.println("start init");
        // 加载配置文件
        doLoadConfig(config.getInitParameter("contextconfiglocation"));
        // 扫描所有包下的类
        doScanner(contextconfig.getProperty("scanpackage"));
        // 对象进行实例化并存储到HashMap中
        doinstance();
        // 根据请求的对象的自动装配
        doautowired();
        // 根据将请求的URL与对应的处理方法结合起来
        doinithandlermapping();
        System.out.println("Spring Framework is init!!");
    }

    private void doLoadConfig(String configlocation){
        InputStream is =  this.getClass().getClassLoader().getResourceAsStream(configlocation);
        try {
            contextconfig.load(is);
        }catch (IOException e){
            e.printStackTrace();
        }finally {
            if(null==is){
                System.out.println("is is NUll");
            }
        }

    }
    private void doautowired(){
        if(ioc.isEmpty()) return;
        for(Map.Entry<String,Object> s : ioc.entrySet()){
            Field[] fields = s.getValue().getClass().getDeclaredFields();
            for (Field field:fields){
                if(!field.isAnnotationPresent(MyAutowired.class))continue;
                MyAutowired my = field.getAnnotation(MyAutowired.class);
                String beanName = my.value().trim();
                if("".equals(beanName)){
                    beanName = field.getType().getName();
                }
                field.setAccessible(true);

                try {
                    field.set(s.getValue(),ioc.get(beanName));
                }catch (Exception e){
                    e.printStackTrace();
                    continue;
                }

            }
        }
    }
    private void doinithandlermapping(){
        if(ioc.isEmpty()) return;
        for(Map.Entry<String ,Object>entry:ioc.entrySet()){
            Class<?> clazz = entry.getValue().getClass();
            if(!clazz.isAnnotationPresent(MyController.class))continue;
            String baseurl = "";
            if(clazz.isAnnotationPresent(MyRequestMapping.class)){
                MyRequestMapping requestMapping = clazz.getAnnotation(MyRequestMapping.class);
                baseurl = requestMapping.value();
            }
            for(Method method:clazz.getMethods()){
                if(!method.isAnnotationPresent(MyRequestMapping.class)) continue;
                MyRequestMapping requestMapping = method.getAnnotation(MyRequestMapping.class);
                String url = ("/"+baseurl+"/"+requestMapping.value().replaceAll("/+","/"));
                handlerMapping.put(url,method);
                System.out.println("Mapped"+url+","+method);
            }
        }
    }
    private void doScanner(String s){
        URL url = this.getClass().getClassLoader().getResource("/"+s.replaceAll("\\.","/"));
        File classpath = new File(url.getFile());
        for (File file:classpath.listFiles()){
            if(file.isDirectory()){
                doScanner(s+"."+file.getName());
            }else{
                if(!file.getName().endsWith(".class")) continue;
                String className = s + "." + file.getName().replaceAll(".class","");
                classNames.add(className);
            }
        }
    }
    private String tolowerfirst(String s){
        char[] chars= s.toCharArray();
        chars[0]+=32;
        return String.valueOf(chars);
    }
    private void doinstance(){
        if(classNames.isEmpty()) return;
        try {
            for(String classname:classNames){
                Class<?> clazz = Class.forName(classname);
                if(clazz.isAnnotationPresent(MyController.class)){
                    String beanName = tolowerfirst(clazz.getSimpleName());
                    Object instance = clazz.newInstance();
                    ioc.put(beanName,instance);
                }else if(clazz.isAnnotationPresent(MyService.class)){
                    String beanName = tolowerfirst(clazz.getSimpleName());
                    MyService service = clazz.getAnnotation(MyService.class);
                    if("".equals(service.value())){
                        beanName = service.value();
                    }
                    Object instance = clazz.newInstance();
                    ioc.put(beanName,instance);
                    for(Class<?> i :clazz.getInterfaces()){
                        if(ioc.containsKey(i.getName())){
                            throw new Exception("this beanName is error");
                        }
                        ioc.put(i.getName(),instance);
                    }
                }else{
                    continue;
                }
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
```