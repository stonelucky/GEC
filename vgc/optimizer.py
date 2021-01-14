import tensorflow as tf
from keras import backend as K
from AMSGrad import AMSGrad

flags = tf.app.flags
FLAGS = flags.FLAGS

class OptimizerVGC(object):
    def __init__(self, preds, labels, preds_X, labels_X, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        preds_X_sub = preds_X
        labels_X_sub = labels_X

        # Define the structure recovery loss
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        # Define the attribute (content) recovery loss
        self.cost += norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_X_sub, targets=labels_X_sub, pos_weight=pos_weight))
        # Define the cat_loss (second item in our paper) and the kl_loss (third item in our paper)
        z_mean = K.expand_dims(model.z_mean, 1)
        z_log_var = K.expand_dims(model.z_log_std, 1)
        y = model.y
        z_prior_mean = model.z_prior_mean

        kl_loss = - 0.5 * (z_log_var - K.square(z_prior_mean))
        kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0) / num_nodes
        cat_loss = K.mean(y * K.log(y + K.epsilon()), 0) / num_nodes
        
        # Final loss
        self.cost = K.sum(self.cost) + K.sum(kl_loss)  + K.sum(cat_loss)

        self.cost = tf.clip_by_value(self.cost, 1e-10, 1e100)
        
        self.optimizer = AMSGrad(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-8)
        self.opt_op = self.optimizer.minimize(self.cost)

        self.grads_vars = self.optimizer.compute_gradients(self.cost)

class OptimizerVGCG(object):
    def __init__(self, preds, labels, preds_X, labels_X, model, num_nodes, pos_weight, norm, d_real, d_fake):
        preds_sub = preds
        labels_sub = labels
        preds_X_sub = preds_X
        labels_X_sub = labels_X

        # Discrimminator Loss
        dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        self.dc_loss = dc_loss_fake + dc_loss_real

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))
        # Define the structure recovery loss
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        # Define the attribute (content) recovery loss
        self.cost += norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_X_sub, targets=labels_X_sub, pos_weight=pos_weight))
        # Define the cat_loss (second item in our paper) and the kl_loss (third item in our paper)
        z_mean = K.expand_dims(model.z_mean, 1)
        z_log_var = K.expand_dims(model.z_log_std, 1)
        y = model.y
        z_prior_mean = model.z_prior_mean

        kl_loss = - 0.5 * (z_log_var - K.square(z_prior_mean))
        kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0) / num_nodes
        cat_loss = K.mean(y * K.log(y + K.epsilon()), 0) / num_nodes
        
        # Final loss
        self.cost = K.sum(self.cost) + K.sum(kl_loss)  + K.sum(cat_loss)

        self.generator_loss = self.generator_loss + self.cost # add to keep the same as AE

        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.op.name]
        en_var = [var for var in all_variables if 'e_' in var.op.name]


        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                                  beta1=0.9, name='adam1').minimize(self.dc_loss, var_list=dc_var)#minimize(dc_loss_real, var_list=dc_var)

            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                              beta1=0.9, name='adam2').minimize(self.generator_loss,
                                                                                                var_list=en_var)
        self.cost = tf.clip_by_value(self.cost, 1e-10, 1e100)
        
        self.optimizer = AMSGrad(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-8)
        self.opt_op = self.optimizer.minimize(self.cost)

        self.grads_vars = self.optimizer.compute_gradients(self.cost)