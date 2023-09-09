---
title: Spring启动流程
date: 2023-09-09 20:00:00
categories:
  - JAVA
  - Spring
tags:
  - 源码解析
author: Fanrencli
---

### 启动流程

- Spring启动通常由我们自己的项目中main方法调用`SpringApplication.run(XXX.class,args)`作为启动入口

#### Spring中启动源码的解析

```java
// 程序中run方法调用处，在这个方法中创建一个SpringApplication对象
public static ConfigurableApplicationContext run(Class<?>[] primarySources, String[] args) {
		return new SpringApplication(primarySources).run(args);
	}
// 创建SpringApplication对象的方法中，会对此对象中一些属性进行初始化，主要时准备一些资源，
public SpringApplication(ResourceLoader resourceLoader, Class<?>... primarySources) {
		this.resourceLoader = resourceLoader;
		Assert.notNull(primarySources, "PrimarySources must not be null");
		this.primarySources = new LinkedHashSet<>(Arrays.asList(primarySources));
        // 判断启动项目时普通WebServlet项目还是其他例如WebFlux项目，本项目通常为WebServlet
		this.webApplicationType = WebApplicationType.deduceFromClasspath();
        // 创建一些ApplicationContextInitializer的工厂类，并将最终的结果打包为一个集合对象放到SpringApplication对象属性中
		setInitializers((Collection) getSpringFactoriesInstances(ApplicationContextInitializer.class));
        // 创建ApplicationListener的工厂类，用于创建监听线程
		setListeners((Collection) getSpringFactoriesInstances(ApplicationListener.class));
        // 根据JVM的栈跟踪找到我们项目中的main方法（根据名字匹配，因为所有的语言的入口都是main方法，所以必然可以找到），找到方法之后获取这个方法的类名，然后反射创建项目的启动类
		this.mainApplicationClass = deduceMainApplicationClass();
	}
// 初始化SpringApplication类之后，就运行下面的run方法
public ConfigurableApplicationContext run(String... args) {
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		ConfigurableApplicationContext context = null;
		Collection<SpringBootExceptionReporter> exceptionReporters = new ArrayList<>();
		configureHeadlessProperty();
        // 由于spring通常时一个web项目，所以启动监听线程来保证程序的持续运行，这个监听类就是之前创建对象时初始化来的
		SpringApplicationRunListeners listeners = getRunListeners(args);
		listeners.starting();
		try {
			ApplicationArguments applicationArguments = new DefaultApplicationArguments(args);
            // 准备一些环境信息，通常启动一个java程序是通过：java -classpth xxx -jar xx.jar,启动命令中包含了所有的环境信息，由jdk环境，需要加载的jar包的地址，以及其他属性，将这些属性绑定到springapplication项目的属性中。
			ConfigurableEnvironment environment = prepareEnvironment(listeners, applicationArguments);
			configureIgnoreBeanInfo(environment);
			Banner printedBanner = printBanner(environment);
            // 初始化上下文，其实就是一个web容器，一般来说是AnnotationConfigServletWebServerApplicationContext"
            // 在创建web容器的时候，由于AnnotationConfigServletWebServerApplicationContext继承了ServletWebServerApplicationContext继承了GenericWebApplicationContext继承了GenericApplicationContext，在GenericApplicationContext实例化的时候创建了一个对象DefaultListableBeanFactory，而再根据继承关系，DefaultListableBeanFactory->AbstractAutowireCapableBeanFactory->AbstractBeanFactory->FactoryBeanRegistrySupport->DefaultSingletonBeanRegistry,最后在DefaultSingletonBeanRegistry对象中创建了三级缓存的HashMap
            // 所以在初始化web容器的时候就已经创建了BeanFactory，为接下来的Bean创建做准备，可以通过getBeanFactory()获取
            // 此外在初始化web容器的时候，再BeanDefinition中给了几个必须使用的初始化Bean，例如一些配置类
			context = createApplicationContext();
            // 再创建一些异常处理的工厂类，如果出现了异常则通过spring的异常类处理
			exceptionReporters = getSpringFactoriesInstances(SpringBootExceptionReporter.class,
					new Class[] { ConfigurableApplicationContext.class }, context);
            // 将一些Bean注册到一级缓存中，同时加载一些资源
			prepareContext(context, environment, listeners, applicationArguments, printedBanner);
            // 此步骤最重要，
			refreshContext(context);
            // 基本完成了所有的启动流程，下面就是一些日志的输出
			afterRefresh(context, applicationArguments);
			stopWatch.stop();
			if (this.logStartupInfo) {
				new StartupInfoLogger(this.mainApplicationClass).logStarted(getApplicationLog(), stopWatch);
			}
			listeners.started(context);
            // 调用所有实现了CommandLine接口的方法
			callRunners(context, applicationArguments);
		}
		catch (Throwable ex) {
			handleRunFailure(context, ex, exceptionReporters, listeners);
			throw new IllegalStateException(ex);
		}

		try {
			listeners.running(context);
		}
		catch (Throwable ex) {
			handleRunFailure(context, ex, exceptionReporters, null);
			throw new IllegalStateException(ex);
		}
		return context;
	}
```

#### refreshContext方法源码解析

- $refreshContext$最终的方法调用web容器的`refresh`方法

```java
public void refresh() throws BeansException, IllegalStateException {
		synchronized (this.startupShutdownMonitor) {
			// Prepare this context for refreshing.
			prepareRefresh();

			// Tell the subclass to refresh the internal bean factory.
            // 返回的beanFactory包含了从xml、@Bean等其他方式定义的Bean属性
			ConfigurableListableBeanFactory beanFactory = obtainFreshBeanFactory();

			// Prepare the bean factory for use in this context.
            // 添加一些BeanPostProcessor用于责任链模式，将一些环境类和上下文也注册到一级缓存中，方便这些类也可以使用自动注入
            // Aware接口也是在这里进行注册的，我们可以通过实现这写接口的方法，然后让spring启动过程中进行操作（BeanNameAware, BeanFactoryAware, BeanClassLoaderAware）
            // 还注册了ApplicationListenerDetector,用于在启动的时候监听是否有此类的实现类从而调用方法
			prepareBeanFactory(beanFactory);

			try {
				// Allows post-processing of the bean factory in context subclasses.
                // 这个方法在父类中时空实现，留给子类实现的一个方法，可以让子类添加一些BeanPostProcessor，和监听接口，主要功能和上一个方法相似
				postProcessBeanFactory(beanFactory);

				// Invoke factory processors registered as beans in the context.
                // 调用BeanFactory的后处理器，用于添加一些开发者希望创建的一些Bean（通过直接形式声明的，因为之前的Bean全都是资源文件里指定的），后处理器主要包括ConfigurationClassPostProcessor（解析 @Configuration、@Bean、@Import、@PropertySource 等）、PropertySourcesPlaceHolderConfigurer（替换 BeanDefinition 中的 ${ }）、MapperScannerConfigurer(补充 Mapper 接口对应的 BeanDefinition)等，
                // 进入此方法首先会去BeanDefinition中寻找是否存在BeanDefinitionRegistryPostProcessor类的子类，然后找到在createApplicationContext()方法中添加的ConfigurationClassPostProcessor类，然后创建一个类的实例并调用这个类的postProcessBeanDefinitionRegistry方法，然后集中处理@Configuration、@Bean、@Import、@PropertySource注解类。因此首先从BeanDefinition找到main方法，解析main方法的时候需要判断是否有@ComponetScan（必有），根据这个注解扫描路径下的所有类，然后再循环判断扫描到的类是否还有@ComponentScan,直到所有@ComponentScan的注解类都处理完毕之后，我们就可以获得@ComponentScan注解中配置的所有需要扫描的路径，针对每个路径下的所有类，判断这个类是否有@Component注解，筛选得到所有类,注册到BeanDefinition以便后续步骤根据BeanDefinition来初始化Bean。
				invokeBeanFactoryPostProcessors(beanFactory);

				// Register bean processors that intercept bean creation.
                // 这一步是继续从 beanFactory 中找出 bean 后处理器，添加至 beanPostProcessors 集合中
                // bean 后处理器，充当 bean 的扩展点，可以工作在 bean 的实例化、依赖注入、初始化阶段，常见的有：
                // AutowiredAnnotationBeanPostProcessor 功能有：解析 @Autowired，@Value 注解
                // CommonAnnotationBeanPostProcessor 功能有：解析 @Resource，@PostConstruct，@PreDestroy
                // AnnotationAwareAspectJAutoProxyCreator 功能有：为符合切点的目标 bean 自动创建代理
                // 由上一步骤，已经获取了所有需要spring管理的Bean，这个步骤主要为Bean完善需要使用到的BeanPostProcessor,例如是否需要动态代理，是否需要自动注入，以及其他自定义实现的后处理方法，并排序，按照责任链的形式进行调用
				registerBeanPostProcessors(beanFactory);

				// Initialize message source for this context.
                // 主要用于实现国际化功能，如果BeanFactory中由实现接口，则开启，否则返回空
				initMessageSource();

				// Initialize event multicaster for this context.
                // spring提供的了事件发布和监听的功能，此方法用于为ApplicationContext上下文绑定事件广播成员，通常再BeanFactory中找是否存在此类Bean，如果没有则创建一个默认的Bean，此后可以通过ApplicationContext.publishEvent(广播成员)来广播事件
				initApplicationEventMulticaster();

				// Initialize other special beans in specific context subclasses.
                // 这一步是空实现，留给子类扩展
                // SpringBoot 中的子类在这里准备了 WebServer，即内嵌 web 容器
                // 体现的是模板方法设计模式
                // 此方法由于之前已经将由@Configuration注解的类都加载到内存中，如果引入了tomcat则已经再内存了，直接初始化即可
				onRefresh();

				// Check for listener beans and register them.
                // 上面绑定了事件广播成员，此步骤就是注册事件监听成员，可以自定以实现
				registerListeners();

				// Instantiate all remaining (non-lazy-init) singletons.
                // 这一步会将 beanFactory 的成员补充完毕，并初始化所有非延迟单例 bean
                // conversionService 也是一套转换机制，作为对 PropertyEditor 的补充,用于http请求的参数转换
                // embeddedValueResolvers 即内嵌值解析器，用来解析 @Value 中的 ${ }，借用的是 Environment 的功能
                // singletonObjects 即单例池，缓存所有单例对象
                // 对象的创建都分三个阶段，每一阶段都有不同的 bean 后处理器参与进来，扩展功能
				finishBeanFactoryInitialization(beanFactory);

				// Last step: publish corresponding event.
                // 这一步会为 ApplicationContext 添加 lifecycleProcessor 成员，用来控制容器内需要生命周期管理的 bean
                // 如果容器中有名称为 lifecycleProcessor 的 bean 就用它，否则创建默认的生命周期管理器
                // 准备好生命周期管理器，就可以实现
                // 调用 context 的 start，即可触发所有实现 LifeCycle 接口 bean 的 start
                // 调用 context 的 stop，即可触发所有实现 LifeCycle 接口 bean 的 stop
                // 发布 ContextRefreshed 事件，整个 refresh 执行完成
				finishRefresh();
			}

			catch (BeansException ex) {
				if (logger.isWarnEnabled()) {
					logger.warn("Exception encountered during context initialization - " +
							"cancelling refresh attempt: " + ex);
				}

				// Destroy already created singletons to avoid dangling resources.
				destroyBeans();

				// Reset 'active' flag.
				cancelRefresh(ex);

				// Propagate exception to caller.
				throw ex;
			}

			finally {
				// Reset common introspection caches in Spring's core, since we
				// might not ever need metadata for singleton beans anymore...
				resetCommonCaches();
			}
		}
	}
```
