# 实验二：KD树数字识别

[TOC]

## 1、KD 树的实现 

### 1.KD树的构建

以课本例3.2为例：

```python
X_train = np.array([[2, 3],
                    [5, 4],
                    [9, 6],
                    [4, 7],
                    [8, 1],
                    [7, 2]])
kd_tree = KDTree(X_train)
```

``` python
{
   "value": [
      7,
      2
   ],
   "index": 5,
   "left_child": {
      "value": [
         5,
         4
      ],
      "index": 1,
      "left_child": {
         "value": [
            2,
            3
         ],
         "index": 0,
         "left_child": null,
         "right_child": null
      },
      "right_child": {
         "value": [
            4,
            7
         ],
         "index": 3,
         "left_child": null,
         "right_child": null
      }
   },
   "right_child": {
      "value": [
         9,
         6
      ],
      "index": 2,
      "left_child": {
         "value": [
            8,
            1
         ],
         "index": 4,
         "left_child": null,
         "right_child": null
      },
      "right_child": null
   }
}
```

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-21-201441.png" alt="3-1-KD-Tree-Demo.png" style="zoom:70%;" />

生成的kd树与课本示例一致。

### 2.KD树的搜索

``` python
# 设置k值
k = 3
# 查找邻近的结点
dists, indices = kd_tree.query(np.array([[3, 4.5]]), k=k)
# 打印邻近结点
print_k_neighbor_sets(X_train, k, indices, dists)
```

``` python
x点的3个近邻点是(4, 7), (5, 4), (2, 3)，距离分别是2.6926, 2.0616, 1.8028
```

### 3.KD树的插入
在（1.）的基础上插入点（8,9）：

``` python
{
   "value": [
      7,
      2
   ],
   "index": 5,
   "left_child": {
      "value": [
         5,
         4
      ],
      "index": 1,
      "left_child": {
         "value": [
            2,
            3
         ],
         "index": 0,
         "left_child": null,
         "right_child": null
      },
      "right_child": {
         "value": [
            4,
            7
         ],
         "index": 3,
         "left_child": null,
         "right_child": null
      }
   },
   "right_child": {
      "value": [
         9,
         6
      ],
      "index": 2,
      "left_child": {
         "value": [
            8,
            1
         ],
         "index": 4,
         "left_child": null,
         "right_child": null
      },
      "right_child": {
         "value": [
            [
               8,
               9
            ]
         ],
         "index": null,
         "left_child": null,
         "right_child": null
      }
   }
}
```

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-21-201450.png" alt="3-1-KD-Tree-Demo.png" style="zoom:70%;" />

添加到（9,6）的右子结点。

### 4.KD 树的删除

将结点（5,4）删除：

``` python
kd_tree = kd_tree.delete_node(point=np.array([[5, 4]]))
```

``` python
{
   "value": [
      7,
      2
   ],
   "index": 5,
   "left_child": {
      "value": [
         2,
         3
      ],
      "index": 0,
      "left_child": null,
      "right_child": {
         "value": [
            4,
            7
         ],
         "index": 3,
         "left_child": null,
         "right_child": null
      }
   },
   "right_child": {
      "value": [
         9,
         6
      ],
      "index": 2,
      "left_child": {
         "value": [
            8,
            1
         ],
         "index": 4,
         "left_child": null,
         "right_child": null
      },
      "right_child": {
         "value": [
            [
               8,
               9
            ]
         ],
         "index": null,
         "left_child": null,
         "right_child": null
      }
   }
}
```

测试过根结点，叶结点和中间结点，结果均正确。

## 2、MNIST 数据集分类 

### 1.直接分类

1、构造MNIST子集，每类取200个点，20%作为测试集。

2、在自己的算法上做测试。

``` python
test accuracy: 0.895
              precision    recall  f1-score   support

         0.0       0.95      0.97      0.96        36
         1.0       0.79      0.98      0.88        47
         2.0       0.97      0.83      0.89        46
         3.0       0.92      0.92      0.92        36
         4.0       0.88      0.82      0.85        34
         5.0       0.94      0.88      0.91        34
         6.0       0.89      0.97      0.93        33
         7.0       0.90      0.84      0.87        44
         8.0       0.98      0.88      0.92        48
         9.0       0.80      0.88      0.84        42

    accuracy                           0.90       400
   macro avg       0.90      0.90      0.90       400
weighted avg       0.90      0.90      0.90       400
```

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-21-201453.png" alt="image-20221126182639787" style="zoom:80%;" />

### 2.PCA降维

利用PCA将数据集的特征由784维，降至25维，分类效果和训练时间有明显的提升。

```python
test accuracy: 0.9375
              precision    recall  f1-score   support

         0.0       0.97      1.00      0.99        36
         1.0       0.92      0.98      0.95        47
         2.0       0.98      0.89      0.93        46
         3.0       0.94      0.94      0.94        36
         4.0       1.00      0.94      0.97        34
         5.0       0.91      0.91      0.91        34
         6.0       0.91      0.97      0.94        33
         7.0       0.93      0.95      0.94        44
         8.0       0.96      0.92      0.94        48
         9.0       0.86      0.88      0.87        42

    accuracy                           0.94       400
   macro avg       0.94      0.94      0.94       400
weighted avg       0.94      0.94      0.94       400
```

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-21-201455.png" alt="image-20221126183121817" style="zoom:80%;" />
