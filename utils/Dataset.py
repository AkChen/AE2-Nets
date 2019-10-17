import h5py
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import scipy.io


class Dataset():
    def __init__(self, name):
        self.path = './dataset/'
        self.name = name

    def load_data(self):
        data_path = self.path + self.name + '.mat'
        if 'ORL' in self.name or 'coil' in self.name:
            dataset = scipy.io.loadmat(data_path)
            x1, x2, y = dataset['x1'], dataset['x2'], dataset['gt']
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
        else:
            dataset = h5py.File(data_path, mode='r')
            x1, x2, y = dataset['x1'], dataset['x2'], dataset['gt']
            x1, x2, y = x1.value, x2.value, y.value
            x1, x2, y = x1.transpose(), x2.transpose(), y.transpose()
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
        return x1, x2, y

    def normalize(self, x, min=0):
        # min_val = np.min(x)
        # max_val = np.max(x)
        # x = (x - min_val) / (max_val - min_val)
        # return x

        if min == 0:
            scaler = MinMaxScaler([0, 1])
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x
