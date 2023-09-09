---
title: SpringBean的生命周期
date: 2023-09-09 20:00:00
categories:
  - JAVA
  - Spring
tags:
  - 源码解析
author: Fanrencli
---

### Bean的生命周期

- Bean的初始化流程：实例化->赋值->初始化（各种postProcessor）
- 实例化之前和之后调用InstantiationAwareBeanPostProcessor接口的postProcessBeforeInstantiation和postProcessAfterInstantiation方法
- 属性赋值之前调用InstantiationAwareBeanPostProcessor接口的postProcessProperties方法
- 属性赋值完成之后进入初始化方法阶段，首先调用Aware的接口来设置相关属性：BeanNameAware/BeanClassLoaderAware/BeanFactoryAware方法
- 然后根据Bean实现的BeanPostProcessor接口顺序调用postProcessBeforeInitialization方法
- 然后开始自身真正的初始化init方法
  - 首先调用InitializingBean的afterPropertiesSet方法
  - 然后调用init-method方法
- 然后再根据Bean实现的BeanPostProcessor接口顺序调用postProcessAfterInitialization方法
- 这样类就初始化完成，可以使用了
- 后面就是类的销毁方法还有一些接口可以实现，如果实现了就需要调用

![Bean生命周期（转载）](http://39.106.34.39:4567/BeanTime.png)

#### finishBeanFactoryInitialization方法源码解析

- $finishBeanFactoryInitialization$最终初始化Bean的代码

```java
@Override
	public void preInstantiateSingletons() throws BeansException {
		if (logger.isTraceEnabled()) {
			logger.trace("Pre-instantiating singletons in " + this);
		}

		// Iterate over a copy to allow for init methods which in turn register new bean definitions.
		// While this may not be part of the regular factory bootstrap, it does otherwise work fine.
		List<String> beanNames = new ArrayList<>(this.beanDefinitionNames);

        //以下的for循环就是初始化Bean的代码，其中只有getBean（）方法我们需要了解
		for (String beanName : beanNames) {
			RootBeanDefinition bd = getMergedLocalBeanDefinition(beanName);
            // 判断是否单例，是否懒加载
			if (!bd.isAbstract() && bd.isSingleton() && !bd.isLazyInit()) {
                // 如果是工厂类
				if (isFactoryBean(beanName)) {
					Object bean = getBean(FACTORY_BEAN_PREFIX + beanName);
					if (bean instanceof FactoryBean) {
						final FactoryBean<?> factory = (FactoryBean<?>) bean;
						boolean isEagerInit;
						if (System.getSecurityManager() != null && factory instanceof SmartFactoryBean) {
							isEagerInit = AccessController.doPrivileged((PrivilegedAction<Boolean>)
											((SmartFactoryBean<?>) factory)::isEagerInit,
									getAccessControlContext());
						}
						else {
							isEagerInit = (factory instanceof SmartFactoryBean &&
									((SmartFactoryBean<?>) factory).isEagerInit());
						}
						if (isEagerInit) {
							getBean(beanName);
						}
					}
				}
				else {
                    // 普通类直接走下面的方法
					getBean(beanName);
				}
			}
		}

        // 截至这里除了懒加载的Bean,所有单例的Bean都初始化完成，下面是如果Bean实现了SmartInitializingSingleton接口就调用方法，由开发者自定义
		// Trigger post-initialization callback for all applicable beans...
		for (String beanName : beanNames) {
			Object singletonInstance = getSingleton(beanName);
			if (singletonInstance instanceof SmartInitializingSingleton) {
				final SmartInitializingSingleton smartSingleton = (SmartInitializingSingleton) singletonInstance;
				if (System.getSecurityManager() != null) {
					AccessController.doPrivileged((PrivilegedAction<Object>) () -> {
						smartSingleton.afterSingletonsInstantiated();
						return null;
					}, getAccessControlContext());
				}
				else {
					smartSingleton.afterSingletonsInstantiated();
				}
			}
		}
	}
protected <T> T doGetBean(final String name, @Nullable final Class<T> requiredType,
			@Nullable final Object[] args, boolean typeCheckOnly) throws BeansException {

		final String beanName = transformedBeanName(name);
		Object bean;

		// Eagerly check singleton cache for manually registered singletons.
		Object sharedInstance = getSingleton(beanName);
        // 如果已经创建了，那么直接返回
		if (sharedInstance != null && args == null) {
			bean = getObjectForBeanInstance(sharedInstance, name, beanName, null);
		}
		else {
			// Fail if we're already creating this bean instance:
			// We're assumably within a circular reference.
            // 此处判断Bean是否已经创建过了，如果创建过了，那么就是产生了循环依赖，报错
			if (isPrototypeCurrentlyInCreation(beanName)) {
				throw new BeanCurrentlyInCreationException(beanName);
			}
			try {
				final RootBeanDefinition mbd = getMergedLocalBeanDefinition(beanName);
				checkMergedBeanDefinition(mbd, beanName, args);

				// Guarantee initialization of beans that the current bean depends on.
                // 判断此类的初始化是否依赖其他类
				String[] dependsOn = mbd.getDependsOn();
				if (dependsOn != null) {
					for (String dep : dependsOn) {
						if (isDependent(beanName, dep)) {
							throw new BeanCreationException(mbd.getResourceDescription(), beanName,
									"Circular depends-on relationship between '" + beanName + "' and '" + dep + "'");
						}
						registerDependentBean(dep, beanName);
						try {
							getBean(dep);
						}
						catch (NoSuchBeanDefinitionException ex) {
							throw new BeanCreationException(mbd.getResourceDescription(), beanName,
									"'" + beanName + "' depends on missing bean '" + dep + "'", ex);
						}
					}
				}
                // 初始化Bean，主要看createBean方法
				// Create bean instance.
				if (mbd.isSingleton()) {
					sharedInstance = getSingleton(beanName, () -> {
						try {
							return createBean(beanName, mbd, args);
						}
						catch (BeansException ex) {
							// Explicitly remove instance from singleton cache: It might have been put there
							// eagerly by the creation process, to allow for circular reference resolution.
							// Also remove any beans that received a temporary reference to the bean.
							destroySingleton(beanName);
							throw ex;
						}
					});
					bean = getObjectForBeanInstance(sharedInstance, name, beanName, mbd);
				}
            }
			catch (BeansException ex) {
				cleanupAfterBeanCreationFailure(beanName);
				throw ex;
			}
		}
	}
    @Override
	protected Object createBean(String beanName, RootBeanDefinition mbd, @Nullable Object[] args)
			throws BeanCreationException {

		if (logger.isTraceEnabled()) {
			logger.trace("Creating instance of bean '" + beanName + "'");
		}
		RootBeanDefinition mbdToUse = mbd;

		// Make sure bean class is actually resolved at this point, and
		// clone the bean definition in case of a dynamically resolved Class
		// which cannot be stored in the shared merged bean definition.
		// 确保 BeanDefinition 中的 Class 被加载
		Class<?> resolvedClass = resolveBeanClass(mbd, beanName);
		if (resolvedClass != null && !mbd.hasBeanClass() && mbd.getBeanClassName() != null) {
			mbdToUse = new RootBeanDefinition(mbd);
			mbdToUse.setBeanClass(resolvedClass);
		}

		// Prepare method overrides.
		// 准备方法覆写，这里又涉及到一个概念：MethodOverrides，它来自于 bean 定义中的 <lookup-method />
		// 和 <replaced-method />，如果读者感兴趣，回到 bean 解析的地方看看对这两个标签的解析。
		try {
			mbdToUse.prepareMethodOverrides();
		}
		catch (BeanDefinitionValidationException ex) {
			throw new BeanDefinitionStoreException(mbdToUse.getResourceDescription(),
					beanName, "Validation of method overrides failed", ex);
		}

		try {
			// Give BeanPostProcessors a chance to return a proxy instead of the target bean instance.
			// 让 InstantiationAwareBeanPostProcessor 在这一步有机会返回代理，如果实现了这接口并且返回了Bean,就不走以下步骤了
			Object bean = resolveBeforeInstantiation(beanName, mbdToUse);
			if (bean != null) {
				return bean;
			}
		}
		catch (Throwable ex) {
			throw new BeanCreationException(mbdToUse.getResourceDescription(), beanName,
					"BeanPostProcessor before instantiation of bean failed", ex);
		}

		try {
			// 重头戏，创建 bean
			Object beanInstance = doCreateBean(beanName, mbdToUse, args);
			if (logger.isTraceEnabled()) {
				logger.trace("Finished creating instance of bean '" + beanName + "'");
			}
			return beanInstance;
		}
		catch (BeanCreationException | ImplicitlyAppearedSingletonException ex) {
			// A previously detected exception with proper bean creation context already,
			// or illegal singleton state to be communicated up to DefaultSingletonBeanRegistry.
			throw ex;
		}
		catch (Throwable ex) {
			throw new BeanCreationException(
					mbdToUse.getResourceDescription(), beanName, "Unexpected exception during bean creation", ex);
		}
	}
protected Object doCreateBean(String beanName, RootBeanDefinition mbd, @Nullable Object[] args)
			throws BeanCreationException {

		// Instantiate the bean.
		BeanWrapper instanceWrapper = null;
		if (mbd.isSingleton()) {
			instanceWrapper = this.factoryBeanInstanceCache.remove(beanName);
		}
		if (instanceWrapper == null) {
			// 如果进入这里说明不是 FactoryBean，这里实例化 Bean，之后进行赋值，然后初始化
			instanceWrapper = createBeanInstance(beanName, mbd, args);
		}
		Object bean = instanceWrapper.getWrappedInstance();
		Class<?> beanType = instanceWrapper.getWrappedClass();
		if (beanType != NullBean.class) {
			mbd.resolvedTargetType = beanType;
		}

		// Allow post-processors to modify the merged bean definition.
		synchronized (mbd.postProcessingLock) {
			if (!mbd.postProcessed) {
				try {
					applyMergedBeanDefinitionPostProcessors(mbd, beanType, beanName);
				}
				catch (Throwable ex) {
					throw new BeanCreationException(mbd.getResourceDescription(), beanName,
							"Post-processing of merged bean definition failed", ex);
				}
				mbd.postProcessed = true;
			}
		}

		// Eagerly cache singletons to be able to resolve circular references
		// even when triggered by lifecycle interfaces like BeanFactoryAware.
		// 下面这块代码是为了解决循环依赖的问题，这是个重头戏，解决循环依赖问题
		boolean earlySingletonExposure = (mbd.isSingleton() && this.allowCircularReferences &&
				isSingletonCurrentlyInCreation(beanName));
		if (earlySingletonExposure) {
			if (logger.isTraceEnabled()) {
				logger.trace("Eagerly caching bean '" + beanName +
						"' to allow for resolving potential circular references");
			}
			addSingletonFactory(beanName, () -> getEarlyBeanReference(beanName, mbd, bean));
		}

		// Initialize the bean instance.
		Object exposedObject = bean;
		try {
			// 这一步也是非常关键的，这一步负责属性装配，因为前面的实例只是实例化了，并没有设值，这里就是设值***************
			populateBean(beanName, mbd, instanceWrapper);
			// 还记得 init-method 吗？还有 InitializingBean 接口？还有 BeanPostProcessor 接口？
			// 这里就是处理 bean 初始化完成后的各种回调**************
			exposedObject = initializeBean(beanName, exposedObject, mbd);
		}
		catch (Throwable ex) {
			if (ex instanceof BeanCreationException && beanName.equals(((BeanCreationException) ex).getBeanName())) {
				throw (BeanCreationException) ex;
			}
			else {
				throw new BeanCreationException(
						mbd.getResourceDescription(), beanName, "Initialization of bean failed", ex);
			}
		}
	// 下面这块代码是为了解决循环依赖的问题，这是个重头戏，解决循环依赖问题	
		if (earlySingletonExposure) {
				//循环依赖的核心方法调用
			Object earlySingletonReference = getSingleton(beanName, false);
			if (earlySingletonReference != null) {
				if (exposedObject == bean) {
					exposedObject = earlySingletonReference;
				}
				else if (!this.allowRawInjectionDespiteWrapping && hasDependentBean(beanName)) {
					String[] dependentBeans = getDependentBeans(beanName);
					Set<String> actualDependentBeans = new LinkedHashSet<>(dependentBeans.length);
					for (String dependentBean : dependentBeans) {
						if (!removeSingletonIfCreatedForTypeCheckOnly(dependentBean)) {
							actualDependentBeans.add(dependentBean);
						}
					}
					if (!actualDependentBeans.isEmpty()) {
						throw new BeanCurrentlyInCreationException(beanName,
								"Bean with name '" + beanName + "' has been injected into other beans [" +
								StringUtils.collectionToCommaDelimitedString(actualDependentBeans) +
								"] in its raw version as part of a circular reference, but has eventually been " +
								"wrapped. This means that said other beans do not use the final version of the " +
								"bean. This is often the result of over-eager type matching - consider using " +
								"'getBeanNamesForType' with the 'allowEagerInit' flag turned off, for example.");
					}
				}
			}
		}

		// Register bean as disposable.
		try {
			registerDisposableBeanIfNecessary(beanName, bean, mbd);
		}
		catch (BeanDefinitionValidationException ex) {
			throw new BeanCreationException(
					mbd.getResourceDescription(), beanName, "Invalid destruction signature", ex);
		}

		return exposedObject;
	}

```

### 三级缓存解决循环依赖

- 实例化BeanA之后，对属性进行赋值的时候发现赋值的BeanB没有，所以需要初始化BeanB,再BeanA赋值的过程中我们并不知道BeanA之后是否需要代理，如果需要那么此时必须将BeanA的代理工厂类放到三级缓存中，防止真的需要代理，此时初始化BeanB发现依赖BeanA,则去三级缓存中找。先找一级缓存，没有就二级缓存，再没有就三级缓存拿到代理类同时将代理类放到二级缓存中，清除三级缓存的工厂类。这样就不怕后面BeanA会代理实现，此时BeanB拿到BeanA完成初始化之后，将BeanB放入以及缓存并返回给BeanA使用。此时BeanA就可以继续赋值，最后需要生成代理类时，判断一下这个类是否是代理类即可，最后完成初始化。