from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, Gaussian
import tensorflow as tf
import keras as keras
from keras.layers import concatenate



flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class VGC(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, num_classes, cat, **kwargs):
        super(VGC, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.num_classes = num_classes
        self.cat = cat
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder'):
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(self.inputs)
            self.hidden2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_2')(self.hidden1)                                      
            self.hidden3 = GraphConvolution(input_dim=FLAGS.hidden2,
                                           output_dim=FLAGS.hidden3,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_3')(self.hidden2)

            if self.cat == True:
                self.merge3 = concatenate([self.hidden1,self.hidden2,self.hidden3], axis = 1)

                self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1 + FLAGS.hidden2 + FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            adj=self.adj,
                                            act=lambda x: x,
                                            dropout=self.dropout,
                                            logging=self.logging,
                                            name='e_dense_4')(self.merge3)

                self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1 + FLAGS.hidden2 + FLAGS.hidden3,
                                                output_dim=FLAGS.hidden4,
                                                adj=self.adj,
                                                act=lambda x: x,
                                                dropout=self.dropout,
                                                logging=self.logging,
                                                name='e_dense_5')(self.merge3)
            if self.cat == False:
                self.z_mean = GraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            adj=self.adj,
                                            act=lambda x: x,
                                            dropout=self.dropout,
                                            logging=self.logging,
                                            name='e_dense_4')(self.hidden3)

                self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden3,
                                                output_dim=FLAGS.hidden4,
                                                adj=self.adj,
                                                act=lambda x: x,
                                                dropout=self.dropout,
                                                logging=self.logging,
                                                name='e_dense_5')(self.hidden3)

            self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden4]) * tf.exp(self.z_log_std)

            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden4,
                                          act=lambda x: x,
                                          logging=self.logging)(self.z)

            self.X_reconstructions = tf.layers.dense(inputs=self.z, units=self.input_dim, activation=tf.nn.relu)
            self.embeddings = self.z
            
            # add gaussian layer to noise the classes
            gaussian = Gaussian(self.num_classes)
            self.z_prior_mean = gaussian(self.z)
            # output the classes            
            y = GraphConvolution(input_dim=FLAGS.hidden4,
                       output_dim=FLAGS.class1,
                       adj=self.adj,
                       act=lambda x: x,
                       dropout=self.dropout,
                       logging=self.logging)(self.z)
            self.y = tf.layers.dense(inputs=self.z, units=self.num_classes, activation=tf.nn.softmax)

def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        # np.random.seed(1)
        # tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out

class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.act = tf.nn.relu

    def construct(self, inputs, reuse = False):
        # with tf.name_scope('Discriminator'):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # np.random.seed(1)
            tf.set_random_seed(1)
            dc_den1 = tf.nn.relu(dense(inputs, FLAGS.hidden4, FLAGS.hidden3, name='dc_den1'))
            dc_den2 = tf.nn.relu(dense(dc_den1, FLAGS.hidden3, FLAGS.hidden1, name='dc_den2'))
            output = dense(dc_den2, FLAGS.hidden1, 1, name='dc_output')
            return output
            
def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise       
