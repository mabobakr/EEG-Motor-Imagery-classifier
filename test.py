from preprocessing import * 
from sklearn import svm
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

import torch
from torch import nn, optim
from classifier import nn_train, nn_predict, nn_accuracy
from torchvision import datasets, transforms, models
import time

max_train = 0
max_test = 0
best_model = -1
max_of_all_train = -1
max_of_all_test = -1
for i in range(51):
  X_train, X_test, y_train, y_test = read(i)
  print("final shapes are: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


  train_coeff = featurize(X_train)
  coeff_len = len(train_coeff)

  csp = [mne.decoding.CSP() for _ in range(coeff_len)]
  X_train = np.concatenate(tuple(csp[x].fit_transform(train_coeff[x][:,:,:], y_train) for x  in range(coeff_len)),axis=-1)

  test_coeff = featurize(X_test)
  X_test = np.concatenate(tuple(csp[x].transform(test_coeff[x][:,:,:]) for x  in range(coeff_len)),axis=-1)


  # now we have CA6 + CD6 -> CD1 (6 matrices) * 7 features = 42 feature vector
  clf = pipeline.make_pipeline(StandardScaler(), svm.SVC())
  # clf = QuadraticDiscriminantAnalysis()
  clf.fit(X_train, y_train)
  print(X_train.shape)

  train_acc = sum(clf.predict(X_train) == y_train) / len(X_train)
  print("Accuracy is ", train_acc)
  
  test_acc = sum(clf.predict(X_test) == y_test) / len(X_test)
  print("test Accuracy is ", test_acc)

  if train_acc > max_train and test_acc > max_test:
    best_model = i
    max_train = train_acc
    max_test = test_acc
  
  if train_acc > max_of_all_train:
    max_of_all_train = train_acc
  
  if test_acc > max_of_all_test:
    max_of_all_test = test_acc
  

print("best seed is ", i)
print("maximum train accuracy reached of all is ", max_of_all_train)
print("maximum test accuracy reached of all is ", max_of_all_test)
print(f"best model train accuracy is {max_train} and test accuracy is {max_test}")

# c = pywt.WaveletPacket(x[0,0,:], 'db4', mode='symmetric', maxlevel=5)

# for node in c.get_level(5, 'natural'):
  # print(node.path)

