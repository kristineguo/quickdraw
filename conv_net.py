import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from util import *
from model import *

NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_CLASSES = 345
BATCH_SIZE = 32

USE_ALL = 1000000
NUM_TRAINING = 10000#USE_ALL
NUM_VALIDATION = 10000#USE_ALL

def get_scores(logits, y, is_test):
    index_to_category, _ = get_category_mappings()
    pred = defaultdict(list)
    for i in range(logits.shape[0]):
        pred[index_to_category[y[i]]].append([index_to_category[logits[i][j]] for j in range(3)])
    mapk3, acc = compute_scores(pred, verbose=is_test)
    return mapk3, acc/y.shape[0]

def run_epoch(model, sess, x, y, is_training, lr=None, is_test=False):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    num_batches = int((indices.shape[0] + BATCH_SIZE - 1)/BATCH_SIZE)

    avg_loss = 0
    logits = np.zeros((x.shape[0], 3), dtype=np.int32)
    for b in range(num_batches):
        cur_indices = indices[(b*BATCH_SIZE):((b + 1)*BATCH_SIZE)]
        x_batch = x[cur_indices]
        y_batch = y[cur_indices]

        if is_training:
            predictions, loss = model.train_on_batch(sess, x_batch, y_batch, lr)
        else:
            predictions, loss = model.predict_on_batch(sess, x_batch, y_batch)
        for i in range(cur_indices.shape[0]):
            logits[cur_indices[i]] = predictions[i].argsort()[-3:][::-1]
        avg_loss += loss/num_batches

    mapk_score, acc = get_scores(logits, y, is_test)
    if not is_test:
        print("loss:", avg_loss)
        print("MAPK@3:", mapk_score)
        print("Accuracy:", acc)
    return avg_loss, mapk_score

def preprocess(x, y, mean=None, std=None):
    x = x.astype(float)
    y = y.astype(int)
    if mean is None:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0) + 10
    x = (x - mean)/std
    return np.reshape(x, (x.shape[0], 28, 28, 1)), y, mean, std

def plot_charts(save_path, train_loss, val_loss, train_mapk, val_mapk):
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label='train loss')
    plt.plot(range(len(val_loss)), val_loss, label='validation loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('CNN Loss Plots')
    plt.savefig(save_path + 'loss.png')

    plt.figure()
    plt.plot(range(len(train_mapk)), train_mapk, label='train MAPK@3')
    plt.plot(range(len(val_mapk)), val_mapk, label='validation MAPK@3')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MAPK@3')
    plt.title('CNN MAPK@3 Plots')
    plt.savefig(save_path + 'mapk.png')

def run(model, sess, saver, save_path):
    x_train, y_train = load_dataset("train")
    x_val, y_val = load_dataset("val")

    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)

    x_train, y_train, mean, std = preprocess(x_train[indices[:NUM_TRAINING]], y_train[indices[:NUM_TRAINING]])
    x_val, y_val, _, _ = preprocess(x_val[:NUM_VALIDATION], y_val[:NUM_VALIDATION], mean, std)

    train_loss, val_loss = [], []
    train_mapk, val_mapk = [], []
    for epoch in range(NUM_EPOCHS):
        print("Running on Epoch %d..." % epoch)
        print("Training...")
        cur_train_loss, cur_train_mapk = run_epoch(model, sess, x_train, y_train, True, LEARNING_RATE)
        print("Evaluating...")
        cur_val_loss, cur_val_mapk = run_epoch(model, sess, x_val, y_val, False)
        if len(val_mapk) == 0 or max(val_mapk) < cur_val_mapk:
            print("New best validation score! Saving model weights...")
            saver.save(sess, save_path + 'model.weights')
        train_loss.append(cur_train_loss)
        val_loss.append(cur_val_loss)
        train_mapk.append(cur_train_mapk)
        val_mapk.append(cur_val_mapk)
    
    plot_charts(save_path, train_loss, val_loss, train_mapk, val_mapk)

    return mean, std

def test(model, sess, saver, save_path, mean, std):
    print("Restoring best weights...")
    saver.restore(sess, save_path + 'model.weights')

    x_test, y_test = load_dataset("test")
    x_test, y_test, _, _ = preprocess(x_test, y_test, mean, std)
    print("Running model on test set...")
    _, test_mapk = run_epoch(model, sess, x_test, y_test, False, is_test=True)

def main():
    save_path = "weights/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    with tf.Graph().as_default():
        model = Model()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        mean, std = run(model, sess, saver, save_path)
        test(model, sess, saver, save_path, mean, std)

if __name__ == "__main__":
    main()
