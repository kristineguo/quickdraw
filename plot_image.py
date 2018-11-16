import numpy as np
from matplotlib import pyplot as plt

img_array = np.load('aircraft _carrier.npy')
print(img_array.shape)
for i in range(img_array.shape[0]):
    plt.imshow(np.reshape(img_array[i], (28, 28)), cmap='gray')
    plt.show()
