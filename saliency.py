import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import io
from scipy.ndimage.filters import gaussian_filter1d
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
    saliency = np.zeros((X.shape[0], 28, 28))
    for i in range(X.shape[0]):
        x = X[i].reshape((1, 28, 28, 1))
        saliency[i] = compute_saliency_map(model, sess, x, [y[i]]).reshape(28, 28)
    return saliency

def show_saliency_maps(model, sess, X, y, mask):
    mask = np.asarray(mask)
    saliency = compute_saliency_maps(model, sess, X, y)
    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(X[i].astype('uint8').reshape((28, 28, 1)))
        plt.axis('off')
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        print(saliency[i], saliency[i].shape)
        a = saliency[i].reshape(28, 28, 1)
        a = np.lib.pad(a, ((0,0), (0, 0), (0,2)), 'constant', constant_values=(0, 0))
        a *= 255.0 / np.max(a)
        io.imsave('saliency_' + str(i) + '.jpg', a.astype('uint8'))
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()

def get_mean_std(model, sess):
    x_train, y_train = load_dataset("train")
    x_train, y_train, mean, std = preprocess(x_train, y_train)
    return mean, std

def get_saliency_maps(model, sess, saver, save_path, mean, std):
    print("Restoring best weights...")
    saver.restore(sess, save_path + 'model.weights')

    x_test, y_test = load_dataset("test")
    x_test, y_test, _, _ = preprocess(x_test, y_test, mean, std)
    index_to_category, _ = get_category_mappings()
    for i in range(y_test.shape[0]):
        if index_to_category[y[i]] == 'apple':
            show_saliency_maps(model, sess, np.array(x_test[i]), np.array(y[i]), np.arange(1))
            break

def main():
    save_path = "weights/".format(datetime.now())
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
