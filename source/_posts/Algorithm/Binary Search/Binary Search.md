---
title: 算法
top: true
cover: true
categories:
  - Algorithm
tags:
  - C++
  - Binary Search
  - Fanrencli
author: Fanrencli
---
> 二叉树专题（`Binary Search`）
> 二叉树作为算法经典题型，应用范围广，需要及其重视。

## 前序遍历
```cpp
class Solution {
public:
    void Search(TreeNode* root) {
        fun(root);
    }
    void pre_order(TreeNode* root){
        if (root==nullptr) return;
        cout << root->val << endl;
        pre_order(root->left);
        pre_order(root->right);
    }
};
```
## 中序遍历
```cpp
class Solution {
public:
    void Search(TreeNode* root) {
        fun(root);
    }
    void middle_order(TreeNode* root){
        if (root==nullptr) return;
        middle_order(root->left);
        cout << root->val << endl;
        middle_order(root->right);
    }
};
```
## 后序遍历
```cpp
class Solution {
public:
    void Search(TreeNode* root) {
        fun(root);
    }
    void post_order(TreeNode* root){
        if (root==nullptr) return;
        post_order(root->left);
        post_order(root->right);
        cout << root->val << endl;
    }
};
```