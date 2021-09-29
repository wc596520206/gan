import tensorflow as tf
import tensorflow.contrib.layers as tcl


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


class DcGan(object):
    def __init__(self, config):
        self.figure_size = config["model"]["figure_size"]
        self.channel = config["model"]["channel"]
        self.z_dim = config["model"]["z_dim"]


        self.generator_input_size = config["model"]["generator_input_size"]
        self.discriminator_output_filters_num = config["model"]["discriminator_output_filters_num"]

    def d_loss(self, d_real, d_fake):
        """

        :param d_real: 1
        :param d_fake: 0
        :return:
        """
        loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        return loss1 + loss2

    def g_loss(self, d_fake):
        """

        :param d_fake: 希望输出都是1
        :return:
        """
        loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        return loss1

    def solver(self, loss, vars, learning_rate=2e-4):
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=vars)

    def create_place_holder(self):
        x = tf.placeholder(tf.float32, shape=[None, self.figure_size, self.figure_size, self.channel])
        z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        return x,z

    def create_generator(self, input):
        """
        创建生成模型
        :return:
        """
        name = "generator"
        with tf.variable_scope(name) as scope:
            g = tcl.fully_connected(input, self.generator_input_size * self.generator_input_size * 1024,
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=tcl.batch_norm)
            g = tf.reshape(g, (-1, self.generator_input_size, self.generator_input_size, 1024))  # size
            g = tcl.conv2d_transpose(g, 512, 3, stride=2,  # size*2
                                     activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 256, 3, stride=2,  # size*4
                                     activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 128, 3, stride=2,  # size*8
                                     activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02))

            g = tcl.conv2d_transpose(g, 1, 3, stride=2,  # size*16
                                     activation_fn=tf.nn.sigmoid, padding='SAME',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02))

        var_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return g, var_all

    def create_discriminator(self, input, reuse=None):
        """
        判别模型
        :return:
        """
        name = "discriminator"
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.conv2d(input, num_outputs=self.discriminator_output_filters_num, kernel_size=4,
                                # bzx64x64x3 -> bzx32x32x64
                                stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=self.discriminator_output_filters_num * 2, kernel_size=4,
                                # 16x16x128
                                stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=self.discriminator_output_filters_num * 4, kernel_size=4,  # 8x8x256
                                stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=self.discriminator_output_filters_num * 8, kernel_size=4,  # 4x4x512
                                stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.flatten(shared)
            d = tcl.fully_connected(shared, 1, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return d, var
