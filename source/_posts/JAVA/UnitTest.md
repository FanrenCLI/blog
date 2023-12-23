---
title: 单元测试解读
date: 2023-12-22 23:03:11
categories:
  - JAVA
tags:
  - UnitTest
author: Fanrencli
---

## 单元测试

### 多方法联测

```java
public class OrderedTestExecutionListener implements TestExecutionListener {

    private static boolean rollbackAfterAll = false;

    @Override
    public void beforeTestMethod(TestContext testContext) {
        // 在每个测试方法开始前，检查是否需要回滚
        if (rollbackAfterAll) {
            testContext.getApplicationContext().getBean(PlatformTransactionManager.class).resetTransaction();
        }
    }

    @Override
    public void afterTestMethod(TestContext testContext) {
        // 在每个测试方法结束后，记录是否发生了异常
        rollbackAfterAll = testContext.hasFailed();
    }

    @Override
    public void afterTestClass(TestContext testContext) {
        // 在整个测试类结束后，如果标记了需要回滚，则进行回滚
        if (rollbackAfterAll) {
            TransactionDefinition def = new DefaultTransactionDefinition();
            TransactionStatus status = testContext.getApplicationContext()
                    .getBean(PlatformTransactionManager.class)
                    .getTransaction(def);
            testContext.getApplicationContext()
                    .getBean(PlatformTransactionManager.class)
                    .rollback(status);
        }
    }
}


@SpringJUnitConfig
@TestExecutionListeners(value = OrderedTestExecutionListener.class, mergeMode = MergeMode.MERGE_WITH_DEFAULTS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class YourTestClass {

    @Autowired
    private PlatformTransactionManager transactionManager;

    @Test
    @Order(1)
    @Rollback
    public void testMethod1() {
        // your test logic
    }

    @Test
    @Order(2)
    @Rollback
    public void testMethod2() {
        // your test logic
    }

    // add more test methods as needed
}
```