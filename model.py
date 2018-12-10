import tensorflow as tf

class Model(object):
    def add_placeholders(self):
        self.lr = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.int32, [None])
        self.is_training = tf.placeholder(tf.bool)

    def create_feed_dict(self, inputs_batch, is_training, labels_batch=None, lr=None):
        feed_dict = {}
        feed_dict[self.X] = inputs_batch
        feed_dict[self.is_training] = is_training
        if lr is not None:
            feed_dict[self.lr] = lr
        if labels_batch is not None:
            feed_dict[self.y] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        conv = [self.X]
        conv.append(tf.layers.conv2d(inputs=conv[-1], filters=5, kernel_size=[3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu))
        print(conv[-1].get_shape())
        conv.append(tf.layers.conv2d(inputs=conv[-1], filters=5, kernel_size=[3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu))
        print(conv[-1].get_shape())
        conv.append(tf.layers.conv2d(inputs=conv[-1], filters=5, kernel_size=[3, 3], strides=[1, 1], padding="SAME", activation=tf.nn.relu))
        print(conv[-1].get_shape())
        conv.append(tf.nn.max_pool(conv[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))
        print(conv[-1].get_shape())
        pool_flat = tf.reshape(conv[-1], [-1, 14 * 14 * 5])
        h1 = tf.layers.dense(inputs=pool_flat, units=700, activation=tf.nn.relu)
        h1_drop = tf.layers.dropout(inputs=h1, rate=0.2, training=self.is_training)
        h2 = tf.layers.dense(inputs=h1_drop, units=500, activation=tf.nn.relu)
        h2_drop = tf.layers.dropout(inputs=h2, rate=0.2, training=self.is_training)
        h3 = tf.layers.dense(inputs=h2_drop, units=400, activation=tf.nn.relu)
        h3_drop = tf.layers.dropout(inputs=h3, rate=0.2, training=self.is_training)
        pred = tf.layers.dense(inputs=h3_drop, units=345)
        return pred
        
    def add_loss_op(self, pred):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=pred))

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, rate):
        feed = self.create_feed_dict(inputs_batch, True, labels_batch=labels_batch, lr=rate)
        _, pred, loss = sess.run([self.train_op, self.pred, self.loss], feed_dict=feed)
        return pred, loss

    def predict_on_batch(self, sess, inputs_batch, labels_batch=None):
        if labels_batch is None:
            feed = self.create_feed_dict(inputs_batch, False)
            predictions = sess.run(self.pred, feed_dict=feed)
            return predictions
        else:
            feed = self.create_feed_dict(inputs_batch, False, labels_batch=labels_batch)
            pred, loss = sess.run([self.pred, self.loss], feed_dict=feed)
            return pred, loss

    def __init__(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
