import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


class Train(object):
    def __init__(self, config, processData):
        self.epoches = config["model"]["epoches"]
        self.batch_size = config["model"]["batch_size"]
        self.z_dim = config["model"]["z_dim"]
        self.sample_dir = config["model"]["sample_dir"]
        self.learning_rate_generator = config["model"]["learning_rate_generator"]
        self.learning_rate_discriminator = config["model"]["learning_rate_discriminator"]



        self.processData = processData

    def build_model(self, dcgan):
        self.x, self.z = dcgan.create_place_holder()
        self.g_sample, self.g_vars = dcgan.create_generator(self.z)
        self.d_real, self.d_vars = dcgan.create_discriminator(self.x)
        self.d_fake, self.d_vars = dcgan.create_discriminator(self.g_sample, reuse=True)
        self.d_loss = dcgan.d_loss(self.d_real, self.d_fake)
        self.g_loss = dcgan.g_loss(self.d_fake)

        self.d_solver = dcgan.solver(self.d_loss, self.d_vars, self.learning_rate_discriminator)
        self.g_solver = dcgan.solver(self.g_loss, self.g_vars,self.learning_rate_generator)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        pass

    def train(self):
        fig_count = 0
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(self.epoches):
            x, _ = self.processData(self.batch_size)
            self.sess.run(self.d_solver,
                          feed_dict={
                              self.x: x,
                              self.z: sample_z(self.batch_size, self.z_dim)
                          })

            if epoch % 100 == 0:
                for i in range(10):
                    self.sess.run(self.g_solver,
                                  feed_dict={
                                      self.z: sample_z(self.batch_size, self.z_dim)
                                  })

                D_loss_curr = self.sess.run(
                    self.d_loss,
                    feed_dict={self.x: x, self.z: sample_z(self.batch_size, self.z_dim)})
                G_loss_curr = self.sess.run(
                    self.g_loss,
                    feed_dict={self.z: sample_z(self.batch_size, self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

            if epoch % 1000 == 0:
                samples = self.sess.run(self.g_sample, feed_dict={self.z: sample_z(16, self.z_dim)})
                fig = self.processData.data2fig(samples)
                plt.savefig('{}/{}.png'.format(self.sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
                fig_count += 1
                plt.close(fig)
