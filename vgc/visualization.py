from __future__ import division
from __future__ import print_function
from sklearn.cluster import KMeans
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
from metrics import clustering_metrics
from constructor import get_placeholder, get_model, format_data, get_optimizer, update, update_with_gan

from matplotlib.ticker import NullFormatter
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# Settings
# flags = tf.app.flags
# FLAGS = flags.FLAGS

class Visualization_Runner():
    def __init__(self, settings):

        print("Visualizaiton on dataset: %s, model: %s, number of iteration: %3d" % (settings['data_name'], settings['model'], settings['iterations']))

        self.data_name = settings['data_name']
        self.iteration =settings['iterations']
        self.model = settings['model']
        self.n_clusters = settings['clustering_num']
        self.cat = settings['cat']

    def _vis(self, features, labels, data_name, n_clusters, label_flag):
        # Plot our dataset.
        print('Please wait t-SNE to visualize the embeddings, it may cost a few seconds.')

        fig = plt.figure(figsize=(8, 8))
        # plt.suptitle("TSNE visualization on {}".format(data_name), fontsize=14)
        ax = fig.add_subplot(1,1,1)

        colormap = [plt.cm.Set1(co) for co in range(n_clusters+1)]

        tsne = manifold.TSNE(n_components = 2, init='pca')
        trans_data = tsne.fit_transform(features)

        plt.scatter(trans_data[:,0], trans_data[:,1], c=np.asarray(colormap)[labels])

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        if label_flag == True:
            fig.savefig('{}_{}_true_label.pdf'.format(self.data_name, self.model), bbox_inches='tight')
        else:
            fig.savefig('{}_{}_predict_label.pdf'.format(self.data_name, self.model), bbox_inches='tight')
        # plt.show()

    def erun(self):
        model_str = self.model

        # formatted data
        feas = format_data(self.data_name)

        # Define placeholders
        placeholders = get_placeholder(feas['adj'])

        # construct model
        d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'], num_classes=self.n_clusters, cat=self.cat)

        # Optimizer
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])

        # Initialize session
        sess = tf.Session()
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(self.iteration):
            if model_str in ['arga', 'arvga', 'vgcg']:
                emb, avg_loss = update_with_gan(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])
            elif model_str in ['gae', 'vgae', 'vgc']:
                emb, avg_loss = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])
            else:
                print('ERROR: model has not be included!')

            if (epoch+1) % 1 == 0:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=10).fit(emb)
                print("Epoch: {:04d} Loss: {:.4f}".format(epoch + 1, avg_loss))
                predict_labels = kmeans.predict(emb)
                # predict_labels = classes
                cm = clustering_metrics(feas['true_labels'], predict_labels)
                # cm.evaluationClusterModelFromLabel()
                acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore = cm.evaluationClusterModelFromLabel()
                print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))
        self._vis(emb, feas['true_labels'], self.data_name, self.n_clusters, label_flag= True)
        self._vis(emb, predict_labels, self.data_name, self.n_clusters, label_flag= False)