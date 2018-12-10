import numpy as np
from util import *
from matplotlib import pyplot as plt

img_array, labels = load_dataset('train_normalized')
print img_array.shape
for i in range(4):
    r = np.random.randint(0, img_array.shape[0])
    plt.figure()
    print labels[r]
    plt.imshow(np.reshape(img_array[r], (28, 28)), cmap='gray')
    plt.show()
    #plt.savefig('img'+str(i))
