import numpy as np
import json
import MNIST_subset as subset

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt

class Node:
    """节点类"""

    def __init__(self, value, index, left_child, right_child):
        self.value = value.tolist()
        self.index = index
        self.left_child = left_child
        self.right_child = right_child

    def __repr__(self):
        return json.dumps(self, indent=3, default=lambda obj: obj.__dict__, ensure_ascii=False, allow_nan=False)


class KDTree:
    """kd tree类"""

    def __init__(self, data):
        # 数据集
        self.data = np.asarray(data)
        # kd树
        self.kd_tree = None
        # 创建平衡kd树
        self._create_kd_tree(data)

    def _split_sub_tree(self, data, depth=0):
        # 算法3.2第3步：直到子区域没有实例存在时停止
        if len(data) == 0:
            return None
        # 算法3.2第2步：选择切分坐标轴, 从0开始（书中是从1开始）
        axis = depth % data.shape[1]
        # 对数据进行排序
        data = data[data[:, axis].argsort()]
        # 算法3.2第1步：将所有实例坐标的中位数作为切分点
        median_index = data.shape[0] // 2
        # 获取结点在数据集中的位置
        node_index = [i for i, v in enumerate(
            self.data) if list(v) == list(data[median_index])]
        return Node(
            # 本结点
            value=data[median_index],
            # 本结点在数据集中的位置
            index=node_index[0],
            # 左子结点
            left_child=self._split_sub_tree(data[:median_index], depth + 1),
            # 右子结点
            right_child=self._split_sub_tree(data[median_index + 1:], depth + 1)
        )

    def _create_kd_tree(self, X):
        self.kd_tree = self._split_sub_tree(X)

    def query(self, data, k=1):
        data = np.asarray(data)
        hits = self._search(data, self.kd_tree, k=k, k_neighbor_sets=list())
        dd = np.array([hit[0] for hit in hits])  # 近邻点
        ii = np.array([hit[1] for hit in hits])  # 近邻点在数据集中的索引
        return dd, ii

    def __repr__(self):
        return str(self.kd_tree)

    @staticmethod
    def _cal_node_distance(node1, node2):
        """计算两个结点之间的距离"""
        return np.sqrt(np.sum(np.square(node1 - node2)))

    def _search(self, point, tree=None, k=1, k_neighbor_sets=None, depth=0):
        n = point.shape[1]
        if k_neighbor_sets is None:
            k_neighbor_sets = []
        if tree is None:
            return k_neighbor_sets

        # (1)找到包含目标点x的叶结点
        if tree.left_child is None and tree.right_child is None:
            # 更新当前k近邻点集
            return self._update_k_neighbor_sets(k_neighbor_sets, k, tree, point)

        # 递归地向下访问kd树
        if point[0][depth % n] < tree.value[depth % n]:
            direct = 'left'
            next_branch = tree.left_child
        else:
            direct = 'right'
            next_branch = tree.right_child
        if next_branch is not None:
            # (3)(b)检查另一子结点对应的区域是否相交
            k_neighbor_sets = self._search(point, tree=next_branch, k=k, depth=depth + 1,
                                           k_neighbor_sets=k_neighbor_sets)

            # 计算目标点与切分点形成的分割超平面的距离
            temp_dist = abs(tree.value[depth % n] - point[0][depth % n])

            if direct == 'left':
                # 判断超球体是否与超平面相交
                if not (k_neighbor_sets[0][0] < temp_dist and len(k_neighbor_sets) == k):
                    # 如果相交，递归地进行近邻搜索
                    # (3)(a) 判断当前结点，并更新当前k近邻点集
                    k_neighbor_sets = self._update_k_neighbor_sets(k_neighbor_sets, k, tree, point)
                    return self._search(point, tree=tree.right_child, k=k, depth=depth + 1,
                                        k_neighbor_sets=k_neighbor_sets)
            else:
                # 判断超球体是否与超平面相交
                if not (k_neighbor_sets[0][0] < temp_dist and len(k_neighbor_sets) == k):
                    # 如果相交，递归地进行近邻搜索
                    # (3)(a) 判断当前结点，并更新当前k近邻点集
                    k_neighbor_sets = self._update_k_neighbor_sets(k_neighbor_sets, k, tree, point)
                    return self._search(point, tree=tree.left_child, k=k, depth=depth + 1,
                                        k_neighbor_sets=k_neighbor_sets)
        else:
            return self._update_k_neighbor_sets(k_neighbor_sets, k, tree, point)

        return k_neighbor_sets

    def _update_k_neighbor_sets(self, best, k, tree, point):
        # 计算目标点与当前结点的距离
        node_distance = self._cal_node_distance(point, tree.value)
        if len(best) == 0:
            best.append((node_distance, tree.index, tree.value))
        elif len(best) < k:
            # 如果“当前k近邻点集”元素数量小于k
            self._insert_k_neighbor_sets(best, tree, node_distance)
        else:
            # 叶节点距离小于“当前 𝑘 近邻点集”中最远点距离
            if best[0][0] > node_distance:
                best = best[1:]
                self._insert_k_neighbor_sets(best, tree, node_distance)
        return best

    @staticmethod
    def _insert_k_neighbor_sets(best, tree, node_distance):
        """将距离最远的结点排在前面"""
        n = len(best)
        for i, item in enumerate(best):
            if item[0] < node_distance:
                # 将距离最远的结点插入到前面
                best.insert(i, (node_distance, tree.index, tree.value))
                break
        if len(best) == n:
            best.append((node_distance, tree.index, tree.value))

    def insert_node(self, point, tree=None, depth=0):
        n = point.shape[1]
        if tree is None:
            tree = self.kd_tree

        # 递归地向下访问kd树, 找到包含目标点x的叶结点
        if point[0][depth % n] < tree.value[depth % n]:
            direct = 'left'
            next_branch = tree.left_child
        else:
            direct = 'right'
            next_branch = tree.right_child
        if next_branch is not None:
            self.insert_node(point, tree=next_branch, depth=depth + 1)
        else:
            # 找到包含目标点x的叶结点,如果v位于当前根结点的左子树，并且当前根节点的左子树为空，那么当前根结点的左孩子直接设置为v并且回退，
            if direct == 'left':
                tree.left_child = Node(value=point, index=None, left_child=None, right_child=None)
            else:
                tree.right_child = Node(value=point, index=None, left_child=None, right_child=None)
        return self

    def delete_node(self, point, tree=None, parent=None, depth=0):
        n = point.shape[1]
        if tree is None:
            tree = self.kd_tree

        # 找目标点
        if (point == np.array(tree.value)).all():
            # 目标点为叶结点,直接删除
            if tree.left_child is None and tree.right_child is None:
                if parent.left_child.index == tree.index:
                    parent.left_child = None
                else:
                    parent.right_child = None
                return
            # 目标点左子树非空，用左子树中最大的结点来替代，并在左子树上递归删除lmaxNode
            elif tree.left_child is not None:
                lmaxNode = self.find_node(tree.left_child, dimension=depth % n, type='max')
                tree.value = lmaxNode.value
                tree.index = lmaxNode.index
                lmaxpoint = np.array(lmaxNode.value).reshape(1, -1)
                self.delete_node(lmaxpoint, tree=tree.left_child, parent=tree, depth=depth + 1)
            # 目标点右子树非空，用右子树中最小的结点来替代，并在右子树上递归删除rminNode
            else:
                rminNode = self.find_node(tree.right_child, dimension=depth % n, type='min')
                tree.value = rminNode.value
                tree.index = rminNode.index
                rminpoint = np.array(rminNode.value).reshape(1, -1)
                self.delete_node(rminpoint, tree=tree.right_child, parent=tree, depth=depth + 1)

        # 递归地向下访问kd树
        if point[0][depth % n] < tree.value[depth % n]:
            next_branch = tree.left_child
        else:
            next_branch = tree.right_child
        if next_branch is not None:
            self.delete_node(point, tree=next_branch, parent=tree, depth=depth + 1)

        return self

    def find_node(self, tree, dimension, type, first_time=True, extremeNode=None):
        # 用于删除结点，找tree中dimension的最值，最大还是最小由type决定
        if type == 'max':
            if first_time:
                # 初始化一个结点用于存储，maxvalue
                extremeNode = Node(value=np.array(tree.value), index=None, left_child=None, right_child=None)
                extremeNode.value[dimension] = float('-inf')
                first_time = False
            if extremeNode.value[dimension] < tree.value[dimension]:
                extremeNode.value = tree.value
                extremeNode.index = tree.index
            if tree.left_child is not None:
                self.find_node(tree.left_child, dimension, type, first_time, extremeNode)
            if tree.right_child is not None:
                self.find_node(tree.right_child, dimension, type, first_time, extremeNode)
            return extremeNode

        if type == 'min':
            if first_time:
                extremeNode = Node(value=np.array(tree.value), index=None, left_child=None, right_child=None)
                extremeNode.value[dimension] = float('inf')
                first_time = False
            if extremeNode.value[dimension] > tree.value[dimension]:
                extremeNode.value = tree.value
                extremeNode.index = tree.index
            if tree.left_child is not None:
                self.find_node(tree.left_child, dimension, type, first_time, extremeNode)
            if tree.right_child is not None:
                self.find_node(tree.right_child, dimension, type, first_time, extremeNode)
            return extremeNode

    def predict(self, test_img, train_label, k):
        pred_labels = []
        for img in test_img:
            _, indices = self.query(img.reshape(1, -1), k=k)
            pred_labels.append(self.get_label(train_label, indices))
        print('test finish')
        return np.array(pred_labels)

    @staticmethod
    def get_label(train_label, k_neighbor_index):
        # 得到近邻的标签
        k_neighbor_labels = train_label[k_neighbor_index]
        k_neighbor_labels = k_neighbor_labels.flatten().astype(int)
        # 统计每个标签出现的次数
        class_counts = np.bincount(k_neighbor_labels)
        # 找到出现次数最多的类别
        pred_label = np.where(class_counts == np.max(class_counts))
        # 如果有多个，只取小的
        return pred_label[0][0]


# 打印信息
def print_k_neighbor_sets(X_train, k, ii, dd):
    if k == 1:
        text = "x点的最近邻点是"
    else:
        text = "x点的%d个近邻点是" % k

    for i, index in enumerate(ii):
        res = X_train[index]
        if i == 0:
            text += str(tuple(res))
        else:
            text += ", " + str(tuple(res))

    if k == 1:
        text += "，距离是"
    else:
        text += "，距离分别是"
    for i, dist in enumerate(dd):
        if i == 0:
            text += "%.4f" % dist
        else:
            text += ", %.4f" % dist

    print(text)

# # 测试例子
# X_train = np.array([[2, 3],
#                     [5, 4],
#                     [9, 6],
#                     [4, 7],
#                     [8, 1],
#                     [7, 2]])
# kd_tree = KDTree(X_train)
# # 设置k值
# k = 3
# # 查找邻近的结点
# dists, indices = kd_tree.query(np.array([[3, 4.5]]), k=k)
# # 打印邻近结点
# print_k_neighbor_sets(X_train, k, indices, dists)
# print(kd_tree)
# # 测试插入
# kd_tree = kd_tree.insert_node(point=np.array([[8, 9]]))
# print(kd_tree)
# # 测试删除
# kd_tree = kd_tree.delete_node(point=np.array([[5, 4]]))
# print(kd_tree)

# MNIST_subset
train_img, test_img, train_label, test_label = subset.load_mnist_subset(size_of_subset=200)

# kd_tree = KDTree(train_img)
#
# k = 5
# pred_label = kd_tree.predict(test_img=test_img, train_label=train_label, k=k)
# print(pred_label)

# PCA
pca = PCA(n_components=25) #实例化
pca = pca.fit(train_img) #拟合模型
pca_train_img = pca.transform(train_img) #获取新矩阵
pca_test_img = pca.transform(test_img) #获取新矩阵

kd_tree = KDTree(pca_train_img)

k = 5
pred_label = kd_tree.predict(test_img=pca_test_img, train_label=train_label, k=k)
print(pred_label)

print('test accuracy:', accuracy_score(test_label, pred_label))
cmat = confusion_matrix(test_label, pred_label)
plt.matshow(cmat, cmap=plt.cm.Reds)
plt.show()
print(confusion_matrix(test_label, pred_label))
print(classification_report(test_label, pred_label))

