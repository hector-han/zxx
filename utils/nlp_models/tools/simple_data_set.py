# --*-- coding: utf-8 --*--
import numpy as np


class BatchDataSet:
    """
    get batch data of input data
    support oversampling
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size=128, oversampling=None):
        self.batch_size = batch_size
        self.X = X
        self.y = y
        if oversampling:
            for cate, sample_rate in oversampling.items():
                indices = np.where(y == cate)[0]
                tmp_x = np.repeat(self.X[indices], sample_rate, axis=0)
                tmp_y = np.repeat(self.y[indices], sample_rate, axis=0)
                self.X = np.concatenate((self.X, tmp_x))
                self.y = np.concatenate((self.y, tmp_y))

        self.length = self.X.shape[0]
        self._index = np.arange(self.length)
        self.cur = 0
        self.init_iterator()

    def init_iterator(self):
        np.random.shuffle(self._index)
        self.cur = 0

    def get_next(self):
        if self.cur >= self.length:
            raise IndexError("BatchDataSet reach the end!")
        next_pos = self.cur + self.batch_size
        if next_pos > self.length:
            next_pos = self.length
        pieces = self._index[self.cur: next_pos]
        ret_X, ret_y = self.X[pieces], self.y[pieces]
        self.cur = next_pos
        return ret_X, ret_y