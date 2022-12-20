import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plot_digit(imgdata):
    '''
    Draw mnist image
    :param imgdata: imgdata is a numpy array
    :return:
    '''
    image = imgdata.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()


def load_mnist_subset(size_of_subset=1000):
    mnist = fetch_openml("mnist_784", data_home='./mnist')
    data, label = mnist["data"], mnist["target"]
    # data.shape: (70000, 784) label.shape (70000,)
    print('datatype:', type(data), 'labeltype', type(label))
    print('data.shape:', data.shape, 'label.shape', label.shape)

    mnistimg = np.array(data.values)
    label = np.array(label.values)

    # 将各类分开
    # num0: 6903, num1: 7877, num2: 6990, num3: 7141, num4: 6824, num5: 6313, num6: 6876, num7: 7293, num8: 6825, num9: 6958
    mnist_subset = np.zeros(shape=(10,size_of_subset,784))
    num_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i, num in enumerate(num_labels):
        index = np.squeeze(np.where(label == num))
        num_img = mnistimg[index]
        mnist_subset[i] = num_img[:size_of_subset]
        print('num{i}:{len}'.format(i=i, len=len(num_img)))
        print('num{i}subset:{len}'.format(i=i, len=len(mnist_subset[i])))

    # (30000, 784)
    mnist_subset = mnist_subset.reshape(10 * size_of_subset, 784)
    # 手动创建标签
    label_subset = np.array([0] * size_of_subset + [1] * size_of_subset + [2] * size_of_subset +
                        [3] * size_of_subset + [4] * size_of_subset + [5] * size_of_subset +
                        [6] * size_of_subset + [7] * size_of_subset + [8] * size_of_subset +
                        [9] * size_of_subset).astype(np.uint8)
    label_subset = label_subset.reshape(-1, 1)

    train_img, test_img, train_label, test_label = train_test_split(mnist_subset, label_subset, test_size=0.2, random_state=42)

    # 数据归一化，加快学习过程，防止某些情况下训练过程出现计算溢出
    train_img = train_img.astype(float) / 255.0
    train_label = train_label.astype(float)
    test_img = test_img.astype(float) / 255.0
    test_label = test_label.astype(float)
    print('train.shape', train_img.shape, 'train_label', train_label.shape)

    print('Data generation completed!')
    return train_img, test_img, train_label, test_label
