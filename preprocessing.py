import numpy as np
import pandas as pd
import os
import mne
import re
import pywt
from sklearn.model_selection import train_test_split
from scipy.signal import butter, sosfilt
from joblib import dump, load



def read_file(fileno):
  i = fileno
  x = np.load(f"numpy_data/A0{i}TX.npy")
  x = x[:, :,1:]
  x = np.swapaxes(x, 1, 2)
  y = np.load(f"numpy_data/A0{i}TY.npy")

  return x, y

# fileno is the number of subect
def read_file_sim(fileno): # For Simulator
  labels = {1: "L", 2: "R", 3: "F", 4: "B"}
  i = fileno
  x = np.load(f"numpy_test_data/{i}X.npy")
  y = np.load(f"numpy_test_data/{i}Y.npy")

  y = np.array([labels[i] for i in y])

  return x, y

# X is data of one sample
# model_name is the joblib file name of model
# csp_name is the joblib file name of the csp list
def predict(x, model, csp):
  labels = {1: "L", 2: "R", 3: "F", 4: "B"}
  x = [x]
  # model = load(model_name)
  # csp = load(csp_name)
  test_coeff = featurize(x)
  coeff_len = len(test_coeff)
  
  X_test_f = np.concatenate(tuple(csp[j].transform(test_coeff[j]) for j in range(coeff_len)), axis=-1)
  return labels[model.predict(X_test_f[0:1])[0]]

# X is data of one sample
# model_name is the joblib file name of model
# csp_name is the joblib file name of the csp list
model = load("model.joblib")
csp = load("csp.joblib")
def efficient_predict(x):
  global model
  global csp
  labels = {1: "L", 2: "R ", 3: "F", 4: "B"}
  x = [x]
  # model = load(model_name)
  # csp = load(csp_name)
  test_coeff = featurize(x)
  coeff_len = len(test_coeff)
  
  X_test_f = np.concatenate(tuple(csp[j].transform(test_coeff[j]) for j in range(coeff_len)), axis=-1)
  return labels[model.predict(X_test_f[0:1])[0]]

def featurize(x):
  coeff = pywt.wavedec(x, 'db4', level = 7)
  return coeff
  


# Apply butterworth filter
# band is an array like [4, 17] where 4 is lower limit and 17 upper
def filter(signal, band, fs):
  sos = butter(5, band, 'bandpass', fs=fs, output='sos')
  filtered = sosfilt(sos, signal)
  return filtered

# Extract EEG training data and convert it to numpy
def convert_file(filename, test=False):
  
  # load the gdf file
  data = mne.io.read_raw_gdf(filename)

  # get data in dataframe format
  dataframe = data.to_data_frame()

  # Get the events 
  events = mne.events_from_annotations(data)
  codes = events[1]
  events = events[0]
  # print(events, codes)
  # for ev in events:
  #   print(ev)

  # convert annotations to mne codes
  if test:
    cfilter = np.asarray(['783'])
  else:
    cfilter = np.asarray(['769', '770', '771', '772'])
  lis = np.asarray([codes[i] for i in cfilter])

  # filter for classes
  ev = events[np.in1d(events[:, 2], lis)]

  # extract the samples from events
  x = np.zeros((288, 313, 26))
  y = np.zeros(288)

  values = dataframe.values#filter(dataframe.values.T, [8, 12], 250).T
  
  for point in range(len(ev)):
    x[point] = values[ev[point][0]:ev[point][0]+313]
    y[point] = ev[point][2] - lis[0] + 1
    
  # Create directory for numpy data
  if not os.path.exists("numpy_data"):
    os.mkdir(os.path.join(os.getcwd(), "numpy_data"))

  # get file name without extension or path
  new_name = os.path.splitext(os.path.basename(filename))[0]
  
  # Save data to numpy arrays
  np.save(f"numpy_data/{new_name}X", x)
  if not test:
    np.save(f"numpy_data/{new_name}Y", y)


# Apply convert file to all files in the data directory
def convert_data():
  if os.path.exists("data"):
    datafiles = os.listdir("data")
    
  else:
    print("data directory doesn't exist")
    exit(1)

  for file in datafiles:
    if re.match(r"A0[0-9]T.gdf", file):
      convert_file("data/" +file)
    if re.match(r"A0[0-9]E.gdf", file):
      convert_file("data/" +file, True)


# Splits the data and save test data in a directory
def create_test_data():
  if not os.path.exists("numpy_test_data"):
    print("Please create a folder named numpy_test_data and rerun")
    print("note: you should run convert_data first")
    exit(1)


  for i in range(1, 10):
    x, y = read_file(i)

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 100, test_size = 0.2)

    # Save data to numpy arrays
    np.save(f"numpy_test_data/{i}X", X_test)
    np.save(f"numpy_test_data/{i}Y", y_test)


