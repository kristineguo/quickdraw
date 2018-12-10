from sklearn import svm
import numpy as np
from util import *
from joblib import dump, load
import os
from collections import defaultdict

train_data, train_labels = load_dataset("train")
eval_data, eval_labels = load_dataset("val")
bias_data = np.zeros(train_data.shape)
(train_data - np.mean(train_data, axis = 1)[:,None])/np.std(train_data, axis = 1)[:,None]
eval_data = (eval_data - np.mean(eval_data, axis = 1)[:,None])/np.std(eval_data, axis = 1)[:,None]
if os.path.isfile('svm_lin.joblib'):
    clf = load('svm_lin.joblib')
else:
    clf = svm.LinearSVR()
    clf.fit(train_data, train_labels)
dump(clf, 'svm_lin.joblib')
print(train_data.shape)


predictions = clf.predict(eval_data)
idx_to_category, _ = get_category_mappings()
pred = defaultdict(list)
for i in range(len(eval_labels)):
    result = idx_to_category[int(min(max(predictions[i],0), len(idx_to_category)-1))]
    category = idx_to_category[int(eval_labels[i])]
    pred[category].append([result])
compute_scores(pred)
