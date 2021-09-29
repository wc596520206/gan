import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class ProcesssData(object):
    def __init__(self, config):
        self.batch_size = config["model"]["batch_size"]
        self.data =  input_data.read_data_sets(config["global"]["train_data_file_path"], one_hot=True)

    def __call__(self, batch_size):
        batch_imgs, y = self.data.train.next_batch(batch_size)
        size = 28
        channel = 1
        batch_imgs = np.reshape(batch_imgs, (self.batch_size, size, size, channel))
        return batch_imgs, y

    def data2fig(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        size = 32
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(size,size), cmap='Greys_r')
        return fig