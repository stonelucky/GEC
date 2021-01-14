import tensorflow as tf
import numpy as np
from model import VGC, Discriminator
from optimizer import OptimizerVGC, OptimizerVGCG
import scipy.sparse as sp
from input_data import load_data
import inspect
from preprocessing import preprocess_graph, sparse_to_tuple, mask_test_edges, construct_feed_dict
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(adj):
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj.shape[0], FLAGS.hidden4],
                                            name='real_distribution')

    }

    return placeholders


def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero, num_classes=1, cat=True):
    discriminator = Discriminator()
    d_real = discriminator.construct(placeholders['real_distribution'])
    model = None
    if model_str == 'vgc' or model_str == 'vgcg':
        model = VGC(placeholders, num_features, num_nodes, features_nonzero, num_classes, cat)

    return d_real, discriminator, model


def format_data(data_name):
    # Load data

    adj, features, true_labels = load_data(data_name)


    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + 2 * sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    feas = {}
    feas['adj'] = adj
    feas['num_features'] = num_features
    feas['num_nodes'] = num_nodes
    feas['features_nonzero'] = features_nonzero
    feas['pos_weight'] = pos_weight
    feas['norm'] = norm
    feas['adj_norm'] = adj_norm
    feas['adj_label'] = adj_label
    feas['features'] = features
    feas['true_labels'] = true_labels
    feas['train_edges'] = train_edges
    feas['val_edges'] = val_edges
    feas['val_edges_false'] = val_edges_false
    feas['test_edges'] = test_edges
    feas['test_edges_false'] = test_edges_false
    feas['adj_orig'] = adj_orig

    return feas

def get_optimizer(model_str, model, discriminator, placeholders, pos_weight, norm, d_real,num_nodes):
    if model_str == 'vgc':
        opt = OptimizerVGC(preds=model.reconstructions,
                            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices=False), [-1]),
                            preds_X=model.X_reconstructions,
                            labels_X=tf.reshape(tf.sparse_tensor_to_dense(placeholders['features'],
                                                                        validate_indices=False), [num_nodes, -1]),
                            model=model, num_nodes=num_nodes,
                            pos_weight=pos_weight,
                            norm=norm)
    elif model_str == 'vgcg':
        opt = OptimizerVGCG(preds=model.reconstructions,
                            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices=False), [-1]),
                            preds_X=model.X_reconstructions,
                            labels_X=tf.reshape(tf.sparse_tensor_to_dense(placeholders['features'],
                                                                        validate_indices=False), [num_nodes, -1]),
                            model=model, num_nodes=num_nodes,
                            pos_weight=pos_weight,
                            norm=norm,
                            d_real=d_real,
                            d_fake=discriminator.construct(model.embeddings, reuse=True))
    return opt

def update_with_gan(model, opt, sess, adj_norm, adj_label, features, placeholders, adj):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)
    # classes = sess.run(model.y, feed_dict=feed_dict).argmax(axis=1)

    z_real_dist = np.random.randn(adj.shape[0], FLAGS.hidden4)
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    avg_cost = reconstruct_loss
    # fh = open('loss_recoder.txt', 'a')
    # fh.write('Loss: %f, d_loss: %f, g_loss: %f' % (avg_cost, d_loss, g_loss))
    # fh.write('\n')
    # fh.flush()
    # fh.close()

    return emb, avg_cost

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['dropout']: 0})
    # emb = sess.run(model.z_mean, feed_dict=feed_dict)
    # classes = sess.run(model.y, feed_dict=feed_dict).argmax(axis=1)
    
    for j in range(1):
        _, reconstruct_loss, emb = sess.run([opt.opt_op, opt.cost, model.z_mean], feed_dict=feed_dict)
    avg_cost = reconstruct_loss
    # fh = open('loss_recoder.txt', 'a')
    # fh.write('Loss: %f' % (avg_cost))
    # fh.write('\n')
    # fh.flush()
    # fh.close()

    return emb, avg_cost

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]