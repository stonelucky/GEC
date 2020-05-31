import tensorflow as tf
import numpy as np
flags = tf.app.flags



flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', .5 * 0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')

flags.DEFINE_integer('hidden3', 192, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 32, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('class1', 32, 'Number of units in class layer 1.')

flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 100, 'number of iterations.')

FLAGS = flags.FLAGS

'''
infor: number of clusters 
'''
infor = {'cora': 7, 'citeseer': 6, 'pubmed':3, 'blog':6, 'flickr': 9, 'cornell': 5, 'texas': 5, 'washington': 5, 'wisconsin': 5}

'''
cat: whether to cat the different layers
'''
cat = True

'''
We set a seed here to report relatively better performance
'''
seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

def get_settings(dataname, model, task):
    if dataname not in infor:
        print('error: wrong data set name')
    if model != 'arga_ae' and model != 'arga_vae' and model != 'arga_cvae':
        print('error: wrong model name')
    if task != 'clustering' and task != 'link_prediction' and task != 'visualization':
        print('error: wrong task name')

    if task == 'clustering' or task == 'visualization':
        iterations = FLAGS.iterations
        clustering_num = infor[dataname]
        re = {'data_name': dataname, 'iterations' : iterations, 'clustering_num' :clustering_num, 'model' : model, 'cat': cat}
    elif task == 'link_prediction':
        iterations = 4 * FLAGS.iterations
        re = {'data_name': dataname, 'iterations' : iterations,'model' : model, 'cat': cat}

    return re
