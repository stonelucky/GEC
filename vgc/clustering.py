from __future__ import division
from __future__ import print_function
from sklearn.cluster import KMeans
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
from metrics import clustering_metrics
from constructor import get_placeholder, get_model, format_data, get_optimizer, update, update_with_gan
# Settings
# flags = tf.app.flags
# FLAGS = flags.FLAGS

class Clustering_Runner():
    def __init__(self, settings):

        print("Clustering on dataset: %s, model: %s, number of iteration: %3d" % (settings['data_name'], settings['model'], settings['iterations']))

        self.data_name = settings['data_name']
        self.iteration =settings['iterations']
        self.model = settings['model']
        self.n_clusters = settings['clustering_num']
        self.cat = settings['cat']

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
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(emb)
                print("Epoch:", '%04d' % (epoch + 1))
                print("Loss:", '%.4f' % avg_loss)
                predict_labels = kmeans.predict(emb)
                
                # predict_labels = classes
                cm = clustering_metrics(feas['true_labels'], predict_labels)
                acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore = cm.evaluationClusterModelFromLabel()
                
                print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

                fh = open('recoder_clustering_{}_{}_iter{}.txt'.format(self.data_name, self.model, str(self.iteration)), 'a')
                fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
                fh.write('\r\n')
                fh.flush()
                fh.close()