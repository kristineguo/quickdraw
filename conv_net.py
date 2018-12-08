import numpy as np
import tensorflow as tf
from util import *
from model import *

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 345
BATCH_SIZE = 32

NUM_TRAINING = 352956
NUM_VALIDATION = 20000

def get_accuracy(pred, y):
    return np.sum(pred == y)/y.shape[0]

def run_epoch(model, sess, x, y, is_training, lr=None):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    num_batches = int((indices.shape[0] + BATCH_SIZE - 1)/BATCH_SIZE)

    avg_loss = 0
    pred = np.zeros(x.shape[0], dtype=np.int32)
    for b in range(num_batches):
        cur_indices = indices[(b*BATCH_SIZE):((b + 1)*BATCH_SIZE)]
        x_batch = x[cur_indices]
        y_batch = y[cur_indices]

        if is_training:
            predictions, loss = model.train_on_batch(sess, x_batch, y_batch, lr)
        else:
            predictions, loss = model.predict_on_batch(sess, x_batch, y_batch)
        for i in range(cur_indices.shape[0]):
            pred[cur_indices[i]] = np.argmax(predictions[i])
        avg_loss += loss/num_batches

        if b % 1000 == 999:
            print("completed %d batches" % (b + 1))
    print("average loss:", avg_loss)
    print("accuracy:", get_accuracy(pred, y))

def preprocess(x, y, mean=None, std=None):
    x = x.astype(float)
    y = y.astype(int)
    if mean is None:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
    x -= mean
    for i in range(std.shape[0]):
        if std[i] > 0:
            x[:, i] /= std[i]
    return np.reshape(x, (x.shape[0], 28, 28, 1)), y, mean, std

def run(model, sess):
    x_train, y_train = load_dataset("train")
    x_val, y_val = load_dataset("val")

    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)

    x_train, y_train, mean, std = preprocess(x_train[indices[:NUM_TRAINING]], y_train[indices[:NUM_TRAINING]])
    x_val, y_val, _, _ = preprocess(x_val[:NUM_VALIDATION], y_val[:NUM_VALIDATION], mean, std)

    for epoch in range(NUM_EPOCHS):
        print("Running on Epoch %d..." % epoch)
        print("Training...")
        run_epoch(model, sess, x_train, y_train, True, LEARNING_RATE)
        print("Evaluating...")
        run_epoch(model, sess, x_val, y_val, False)

def main():
    with tf.Graph().as_default():
        model = Model()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        run(model, sess)

if __name__ == "__main__":
    main()
