import struct
import numpy as np
import gzip
import cv2
import random


def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

"""
dataset for MNIST arithmetic task
"""
class MnistData_arithmetic:
    def __init__(self, batch_size, size=(14, 14), finetune_num=15, seed=0):
        train_data = read_idx('mnist_data/train-images-idx3-ubyte.gz')
        train_labels = read_idx('mnist_data/train-labels-idx1-ubyte.gz')
        val_data = read_idx('mnist_data/t10k-images-idx3-ubyte.gz')
        val_labels = read_idx('mnist_data/t10k-labels-idx1-ubyte.gz')

        train_data_ = np.zeros((train_data.shape[0], size[0], size[1]))
        val_data_ = np.zeros((val_data.shape[0], size[0], size[1]))

        self.training_num = train_data.shape[0]
        self.val_num = val_data.shape[0]
        self.size = size
        self.finetune_num = finetune_num

        for i in range(train_data.shape[0]):
            img = train_data[i, :]
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
            _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

            train_data_[i, :] = img

        for i in range(val_data.shape[0]):
            img = val_data[i, :]
            img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
            _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
            val_data_[i, :] = img

        train_data = train_data_
        val_data = val_data_

        del train_data_

        train_labels_idx, val_labels_idx = [], []
        for i in range(10):
            idx_array = np.where(train_labels == i)[0]
            train_labels_idx.append(idx_array)

        for i in range(10):
            idx_array = np.where(val_labels == i)[0]
            val_labels_idx.append(idx_array)

        """
        Construct the synthetic data and labels
        """
        self.set_random_seed(seed)
        train_select_pairs, test_select_pairs = self.read_pairs()

        train_syn_data, train_syn_label = self.construct_synthetic_data(60000, train_select_pairs, train_labels_idx, train_data)
        val_pair_data, val_pair_label = self.construct_synthetic_data(40000, test_select_pairs, val_labels_idx, val_data, return_each_pair=True)
        finetune_data, finetune_label, test_syn_data, test_syn_label = self.split_test_data(val_pair_data, val_pair_label)

        self.train_data = [train_syn_data[i:i + batch_size] for i in range(0, train_syn_data.shape[0], batch_size)]
        self.train_labels = [train_syn_label[i:i + batch_size] for i in range(0, train_syn_label.shape[0], batch_size)]

        # for few-shot adaptation
        self.ft_data = finetune_data
        self.ft_labels = finetune_label

        # for ood test
        self.test_data = [test_syn_data[i:i + 512] for i in range(0, test_syn_data.shape[0], 512)]
        self.test_labels = [test_syn_label[i:i + 512] for i in range(0, test_syn_label.shape[0], 512)]

        # for iid test
        test_iid_data, test_iid_label = self.construct_synthetic_data(20000, train_select_pairs, val_labels_idx, val_data)
        self.test_iid_data = [test_iid_data[i:i + 512] for i in range(0, test_iid_data.shape[0], 512)]
        self.test_iid_label = [test_iid_label[i:i + 512] for i in range(0, test_iid_label.shape[0], 512)]

    def set_random_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        # print("Random Seed is {} in data generation process!".format(seed))

    # only for remove modules
    def test_iid_len(self):
        return len(self.test_iid_label)

    # only for remove modules
    def test_iid_get(self, i):
        return self.test_iid_data[i] / 255, self.test_iid_label[i]

    def read_pairs(self):
        train_path = 'train_select_pairs.txt'
        test_path = 'test_select_pairs.txt'

        train_select_pairs, test_select_pairs = [], []
        with open(train_path, 'r') as f:
            for line in f.readlines():
                d1, d2 = line.strip().split(',')
                train_select_pairs.append((int(d1), int(d2)))

        with open(test_path, 'r') as f:
            for line in f.readlines():
                d1, d2 = line.strip().split(',')
                test_select_pairs.append((int(d1), int(d2)))
        return train_select_pairs, test_select_pairs

    def split_test_data(self, val_pair_data, val_pair_label):
        # val_pair_data: a list contains multiple lists, each list contains the images for a select digits pair
        # val_pair_label: a list contains multiple lists, each list contains the labels for a select digits pair
        num_pairs = len(val_pair_data)

        test_data, finetune_data = [], []
        test_labels, finetune_labels = [], []
        for i in range(num_pairs):
            test_data += val_pair_data[i][:700]
            test_labels += val_pair_label[i][:700]

            finetune_data.append(val_pair_data[i][700:])
            finetune_labels.append(val_pair_label[i][700:])

        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        return finetune_data, finetune_labels, test_data, test_labels

    def construct_synthetic_data(self, num, select_pairs, labels_idx, images, return_each_pair=False):
        num_pairs = len(select_pairs)
        select_pair_data = [[] for _ in range(num_pairs)]
        select_pair_label = [[] for _ in range(num_pairs)]

        synthetic_images = np.zeros((num, self.size[0] * 2, self.size[0] * 2))
        synthetic_labels = np.zeros((num))

        count = 0

        while count < num:
            idx = random.randrange(num_pairs)
            d1, d2 = select_pairs[idx]
            if random.random() > 0.5:  # we will switch d1 with d2 if p > 0.5
                tmp = d1
                d1 = d2
                d2 = tmp

            if d1 + d2 >= 10:
                label = min(d1, d2)
            else:
                label = max(d1, d2)
            if label >= 10 or label < 0:  # label must be [0,9]
                continue
            else:
                d1_idx = labels_idx[d1]
                d2_idx = labels_idx[d2]
                d1_img = images[random.choice(d1_idx)]
                d2_img = images[random.choice(d2_idx)]
                synthetic_images[count][:14, :14] = d1_img
                synthetic_images[count][14:, 14:] = d2_img
                synthetic_labels[count] = label  # start from 0

                select_pair_data[idx].append(synthetic_images[count])
                select_pair_label[idx].append(label)
                count += 1

        if return_each_pair:
            return select_pair_data, select_pair_label

        return synthetic_images, synthetic_labels

    def train_len(self):
        return len(self.train_labels)

    def test_len(self):
        return len(self.test_labels)

    def train_get(self, i):
        return self.train_data[i] / 255, self.train_labels[i]

    def test_get(self, i):
        return self.test_data[i] / 255, self.test_labels[i]

    def ft_get(self):
        # self.ft_data: a list contains multiple lists, each list contains the images for a select digits pair
        # self.ft_label: a list contains multiple lists, each list contains the labels for a select digits pair
        assert self.finetune_num % len(self.ft_data) == 0
        num_each_pair = int(self.finetune_num / len(self.ft_data))

        ft_data, ft_labels = [], []
        for i in range(len(self.ft_data)):
            ft_data += self.ft_data[i][:num_each_pair]
            ft_labels += self.ft_labels[i][:num_each_pair]

        return np.array(ft_data) / 255, np.array(ft_labels)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    data = MnistData_arithmetic(512)
    train_imgs, train_labels = data.train_get(0)
    idx = random.randrange(512)
    img = train_imgs[idx]
    label = train_labels[idx]
    # ft_imgs, ft_labels = data.ft_get()
    # idx = random.randrange(15)
    # img = ft_imgs[idx]
    # label = ft_labels[idx]
    # print(ft_labels)

    plt.imshow(np.uint8(img))
    plt.show()
    print(label)
