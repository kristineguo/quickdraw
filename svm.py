from sklearn import svm
import numpy as np
from util import *
from joblib import dump, load
from sklearn.preprocessing import RobustScaler
import os
from collections import defaultdict

train_size = 1000

train_data, train_labels = load_dataset("train")
eval_data, eval_labels = load_dataset("val")

rbX = RobustScaler()
train_data = rbX.fit_transform(train_data)

indices = np.random.choice(len(train_data), train_size)
if os.path.isfile('svm.joblib'):
    clf = load('svm.joblib')
else:
    clf = svm.SVC(decision_function_shape='ovo', cache_size=7000, gamma=0.01, C=10)
    clf.fit(train_data[indices], train_labels[indices])
dump(clf, 'svm.joblib')


predictions = clf.predict(rbX.transform(eval_data))
idx_to_category, _ = get_category_mappings()
pred = defaultdict(list)
for i in range(len(eval_labels)):
    result = idx_to_category[int(min(max(predictions[i],0), len(idx_to_category)-1))]
    category = idx_to_category[int(eval_labels[i])]
    pred[category].append([result])
compute_scores(pred)
