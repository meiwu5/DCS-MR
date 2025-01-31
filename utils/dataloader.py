import numpy as np
import torch

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, shuffle=True, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        if shuffle:
            self.shuffle()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size *
                              (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                x_i = torch.Tensor(x_i).to('cuda:0', non_blocking=True)
                y_i = torch.Tensor(y_i).to('cuda:0', non_blocking=True)
                yield {'x': x_i, 'y': y_i}
                self.current_ind += 1

        return _wrapper()
    
class Cotinual_learning_DataLoader(object):
    def __init__(self, xs, batch_size, shuffle=True, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        if shuffle:
            self.shuffle()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs = self.xs[permutation]
        self.xs = xs

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size *
                              (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                x_i = torch.Tensor(x_i).to('cuda:0', non_blocking=True)
                yield {'x': x_i}
                self.current_ind += 1

        return _wrapper()
