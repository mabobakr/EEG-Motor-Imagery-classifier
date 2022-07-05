from preprocessing import * 
from sklearn import svm
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.multiclass import OneVsRestClassifier

import torch
from torch import nn, optim
from classifier import nn_train, nn_predict, nn_accuracy
from torchvision import datasets, transforms, models
import time

values = np.zeros((9, 4))

for i in range(1, 10):
  x, y = read_file(i)

  X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 100, test_size = 0.2)

  print("final shapes are: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

  train_coeff = featurize(X_train)
  coeff_len = len(train_coeff)

  csp = [mne.decoding.CSP(8) for _ in range(coeff_len)]
  X_train = np.concatenate(tuple(csp[x].fit_transform(train_coeff[x], y_train) for x  in range(coeff_len)),axis=-1)

  test_coeff = featurize(X_test)
  X_test = np.concatenate(tuple(csp[x].transform(test_coeff[x]) for x  in range(coeff_len)),axis=-1)



  clf = pipeline.make_pipeline(StandardScaler(), svm.SVC())
  # clf = OneVsRestClassifier(svm.SVC())
  # clf = OneVsRestClassifier(clf)
  clf.fit(X_train, y_train)

  values[i - 1][0] = sum(clf.predict(X_train) == y_train) / len(X_train)
  values[i - 1][1] = sum(clf.predict(X_test) == y_test) / len(X_test)
  values[i - 1][2] = cohen_kappa_score(clf.predict(X_train), y_train)
  values[i - 1][3] = cohen_kappa_score(clf.predict(X_test), y_test)


  print("Accuracy is ", sum(clf.predict(X_train) == y_train) / len(X_train))
  print(len(X_train))

  print("test Accuracy is ", sum(clf.predict(X_test) == y_test) / len(X_test))
  print(len(X_test))

  print("kappa score on train is: ", cohen_kappa_score(clf.predict(X_train), y_train))
  print("kappa score on test is: ", cohen_kappa_score(clf.predict(X_test), y_test))

for i in range(1, 10):
  print(f"Subject {i} ", "="*10)
  print("acc_train", values[i-1][0])
  print("acc_test", values[i-1][1])
  print("kappa train", values[i-1][2])
  print("kappa test", values[i-1][3])

print(values[:, 1])
print("average accuracy on test", (np.sum(values[:, 1])- values[5, 1]) / 8)

### Test filtering code
# t = np.linspace(0, 5, 5000, False)

# sig10 = np.sin(2 * np.pi * 10 * t)
# sig20 = np.sin(2 * np.pi * 20 * t)
# sig15 = np.sin(2 * np.pi * 15 * t)

# sig = sig10 + sig20 + sig15

# def plot(t, sig):
#   fig, (ax1) = plt.subplots(1, 1, sharex=True)
#   ax1.plot(t, sig)
#   ax1.set_title('10 Hz and 20 Hz sinusoids')
#   ax1.axis([0, 5, -2, 2])

# plot(t, sig10)
# plot(t, sig15)
# plot(t, sig20)
# plot(t, sig)


# ### Test the nn code
# max_train = 0
# max_test = 0
# best_model = -1
# max_of_all_train = -1
# max_of_all_test = -1
# for i in range(51):
#   X_train, X_test, y_train, y_test = read(i)
#   print("final shapes are: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#   train_coeff = featurize(X_train)
#   coeff_len = len(train_coeff)

#   csp = [mne.decoding.CSP() for _ in range(coeff_len)]
#   X_train = np.concatenate(tuple(csp[x].fit_transform(train_coeff[x][:,:,:], y_train) for x  in range(coeff_len)),axis=-1)

#   test_coeff = featurize(X_test)
#   X_test = np.concatenate(tuple(csp[x].transform(test_coeff[x][:,:,:]) for x  in range(coeff_len)),axis=-1)


#   # now we have CA6 + CD6 -> CD1 (6 matrices) * 7 features = 42 feature vector
#   clf = pipeline.make_pipeline(StandardScaler(), svm.SVC())
#   # clf = QuadraticDiscriminantAnalysis()
#   clf.fit(X_train, y_train)
#   print(X_train.shape)

#   train_acc = sum(clf.predict(X_train) == y_train) / len(X_train)
#   print("Accuracy is ", train_acc)
  
#   test_acc = sum(clf.predict(X_test) == y_test) / len(X_test)
#   print("test Accuracy is ", test_acc)

#   if train_acc > max_train and test_acc > max_test:
#     best_model = i
#     max_train = train_acc
#     max_test = test_acc
  
#   if train_acc > max_of_all_train:
#     max_of_all_train = train_acc
  
#   if test_acc > max_of_all_test:
#     max_of_all_test = test_acc
  

# print("best seed is ", i)
# print("maximum train accuracy reached of all is ", max_of_all_train)
# print("maximum test accuracy reached of all is ", max_of_all_test)
# print(f"best model train accuracy is {max_train} and test accuracy is {max_test}")

# c = pywt.WaveletPacket(x[0,0,:], 'db4', mode='symmetric', maxlevel=5)

# for node in c.get_level(5, 'natural'):
  # print(node.path)

