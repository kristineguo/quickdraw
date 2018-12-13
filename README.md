# Quick, Draw! Doodle Recognition

## About

Doodle recognition has important consequences in computer vision and pattern recognition, especially in relation to the handling of noisy datasets. We build a multi-class classifier to assign hand-drawn doodles from Google's online game Quick, Draw! into 345 unique categories. To do so, we implement and compare multiple variations of k-nearest neighbors and a convolutional neural network. 

## Repo details

### Data processing scripts
- **create_splits.py**: samples and splits complete dataset into training, validation, and test sets.
- **compute_centroids.py**: takes the training data and computes a centroid for each category.
- **centroids/**: folder containing the .npy and .png of the centroids computed from compute_centroids.py.
- **compute_centroids_plus.py**: takes the training data and computes 5 centroids for each category using K-Means.
- **centroids_plus/**: folder containing the .npy and .png of the centroids computed from compute_centroids_plus.py.

### K-Nearest Neighbors (KNN) Models
- **knn.py**: pipeline that trains and evaluates the 1 Closest Centroid (1-CC) model.
- **knn_pp.py**: pipeline that trains and evaluates the K-Means++ KNN model and its variants.
- **k_nearest_neighbor.py**: util functions for KNN.

### Convolutional Neural Net (CNN)
- **conv_net.py**: pipeline that trains and evaluates the CNN model.
- **model.py**: CNN model architecture. 
- **saliency.py**: computes saliency maps for an apple, blueberry, and onion example.

### Miscellaneous
- **categories.txt**: list of the 345 categories in alphabetical order.
- **util.py**: util functions shared across all of the models.
