import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
from skimage import io
from model import *
from util import *
from conv_net import preprocess

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

def compute_saliency_map(model, sess, X, y):
    correct_scores = tf.gather_nd(model.pred, tf.stack((tf.range(X.shape[0]), model.y), axis=1))
    loss = tf.reduce_sum(correct_scores)
    grads_deep = tf.gradients(loss, model.X)[0]
    grads = tf.reduce_max(tf.abs(grads_deep), axis=3)
    feed = model.create_feed_dict(X, False, labels_batch=y)
    saliency = np.array(sess.run(grads, feed_dict = feed))
    return saliency

def compute_saliency_maps(model, sess, X, y):
    saliency = np.zeros((len(X), 28, 28))
    for i in range(len(X)):
        x = X[i].reshape((1, 28, 28, 1))
        saliency[i] = compute_saliency_map(model, sess, x, [y[i]]).reshape(28, 28)
    return saliency

def show_saliency_maps(model, sess, ogX, X, y, mask):
    mask = np.asarray(mask)
    saliency = compute_saliency_maps(model, sess, X, y)
    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(ogX[i])
        plt.axis('off')
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 6)
    plt.savefig('saliency_img.png')

def get_mean_std(model, sess):
    x_train, y_train = load_dataset("train")
    x_train, y_train, mean, std = preprocess(x_train, y_train)
    return mean, std

def get_saliency_maps(model, sess, saver, save_path, mean, std):
    print("Restoring best weights...")
    saver.restore(sess, save_path + 'model.weights')

    x_test, y_test = load_dataset("test")
    x_test_processed, y_test, _, _ = preprocess(x_test, y_test, mean, std)
    x_test = np.reshape(x_test, (-1, 28, 28))

    index_to_category, _ = get_category_mappings()
    ogX, X, y = [], [], []
    apple, onion, blueberry = 0, 0, 0
    for i in range(y_test.shape[0]):
        if index_to_category[y_test[i]] == 'apple':
            if apple == 12:
                ogX.append(x_test[i])
                X.append(x_test_processed[i])
                y.append(y_test[i])
            apple += 1
        elif index_to_category[y_test[i]] == 'onion':
            if onion == 19:
                ogX.append(x_test[i])
                X.append(x_test_processed[i])
                y.append(y_test[i])
            onion += 1
        elif index_to_category[y_test[i]] == 'blueberry':
            if blueberry == 11:
                ogX.append(x_test[i])
                X.append(x_test_processed[i])
                y.append(y_test[i])
            blueberry += 1

    show_saliency_maps(model, sess, ogX, X, y, np.arange(len(X)))

def main():
    save_path = "weights/20181210_014226/".format(datetime.now())
    with tf.Graph().as_default():
        model = Model()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        mean, std = get_mean_std(model, sess)
        get_saliency_maps(model, sess, saver, save_path, mean, std)

if __name__ == "__main__":
    main()
