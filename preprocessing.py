import numpy as np
import pandas as pd
import os
import mne
import re
import pywt
from sklearn.model_selection import train_test_split
from scipy.signal import butter, sosfilt
from joblib import dump, load
from sklearn.utils import shuffle


def read_file(directory, filename):
  x = np.load(f"{directory}/{filename}X.npy")
  y = np.load(f"{directory}/{filename}Y.npy")
  return x, y

# fileno is the number of subject
# def read_file_sim(fileno): # For Simulator
#   labels = {1: "L", 2: "R", 3: "F", 4: "B"}
#   i = fileno
#   x = np.load(f"numpy_test_data/{i}X.npy")
#   y = np.load(f"numpy_test_data/{i}Y.npy")

#   y = np.array([labels[i] for i in y])

#   return x, y

def read_file_sim(directory, filename):
  labels = {1: "L", 2: "R", 3: "F", 4: "B"}
  x = np.load(f"{directory}/{filename}X.npy")
  y = np.load(f"{directory}/{filename}Y.npy")
  y = np.array([labels[i] for i in y])

  return x, y


# X is data of one sample
model = load("model.joblib")
csp = load("csp.joblib")
def predict(x):
  global model
  global csp
  labels = {1: "L", 2: "R", 3: "F", 4: "B"}
  x = [x]
  test_coeff = featurize(x)
  coeff_len = len(test_coeff)
  
  X_test_f = np.concatenate(tuple(csp[j].transform(test_coeff[j]) for j in range(coeff_len)), axis=-1)
  return labels[model.predict(X_test_f[0:1])[0]]

# X is data of one sample
idle_model = load("idle_model.joblib")
idle_csp = load("idle_csp.joblib")
def predict_idle(x):
  global idle_model
  global idle_csp
  
  x = [x]
  test_coeff = featurize(x)
  coeff_len = len(test_coeff)
  
  x_test_f = np.concatenate(tuple(idle_csp[j].transform(test_coeff[j]) for j in range(coeff_len)), axis=-1)
  return idle_model.predict(x_test_f[0:1])[0]


# apply discrete wavelet transform
def featurize(x):
  coeff = pywt.wavedec(x, 'db4', level = 7)
  return coeff  


# Apply butterworth filter
# band is an array like [4, 17] where 4 is lower limit and 17 upper
def filter(signal, band, fs):
  sos = butter(5, band, 'bandpass', fs=fs, output='sos')
  filtered = sosfilt(sos, signal)
  return filtered


def read_gdf(filename):
  # load the gdf file
  data = mne.io.read_raw_gdf(filename)

  # get data samples shape (num_of_samples, num_electrodes)
  values = data.to_data_frame().values[:, 1:]

  # Get the events 
  events = mne.events_from_annotations(data)
  codes = events[1]
  events = events[0]

  return values, codes, events


# Apply convert file to all files in the data directory
def convert_data():
  if os.path.exists("data"):
    datafiles = os.listdir("data")
    
  else:
    print("data directory doesn't exist")
    exit(1)

  for file in datafiles:
    if re.match(r"A0[0-9].gdf", file):
      
      values, codes, events = read_gdf("data/" + file)
      
      # number of samples of one electrode 
      psize = 750

      # Extract idle data points from raw data
      x_idle, y_idle = idle_points(values, events, psize)
      xi_train, xi_test, yi_train, yi_test = train_test_split(x_idle, y_idle, random_state = 42, test_size = 0.2)
      
      # Extract actions data points from raw data
      x_actions, y_actions = action_points(values, codes, events, psize)
      xa_train, xa_test, ya_train, ya_test = train_test_split(x_actions, y_actions, random_state = 42, test_size = 0.2)

      # xa => x of actions model, xi => x of idle model

      # get file name without extension or path
      new_name = os.path.splitext(os.path.basename(file))[0]

        # Create directory for numpy data
      dir_name = "action_train"
      if not os.path.exists(dir_name):
        os.mkdir(os.path.join(os.getcwd(), dir_name))
      
      # save X_actions and y_actions train
      np.save(f"{dir_name}/{new_name}X", xa_train)
      np.save(f"{dir_name}/{new_name}Y", ya_train)

      dir_name = "action_test"
      if not os.path.exists(dir_name):
        os.mkdir(os.path.join(os.getcwd(), dir_name))
      
      # save X_action and y_actions test
      np.save(f"{dir_name}/{new_name}X", xa_test)
      np.save(f"{dir_name}/{new_name}Y", ya_test)

      dir_name = "idle_train"
      if not os.path.exists(dir_name):
        os.mkdir(os.path.join(os.getcwd(), dir_name))
      
      # save (X_actions train + X_idle train) and (y_actions train + y_idle train)
      
      # change y-labels to from 1,2,3,4 to 1 (means action)
      ya_train = np.ones((len(ya_train)))

      x = np.concatenate((xa_train, xi_train))
      y = np.concatenate((ya_train, yi_train))
      x, y = shuffle(x, y)
      np.save(f"{dir_name}/{new_name}X", x)
      np.save(f"{dir_name}/{new_name}Y", y)


      dir_name = "idle_test"
      if not os.path.exists(dir_name):
        os.mkdir(os.path.join(os.getcwd(), dir_name))

      # save (X_actions test + X_idle test) and (y_actions test + y_idle test)
      
      # change y-labels to from 1,2,3,4 to 1 (means action)
      ya_test = np.ones((len(ya_test)))
      x = np.concatenate((xa_test, xi_test))
      y = np.concatenate((ya_test, yi_test))
      x, y = shuffle(x, y)
      np.save(f"{dir_name}/{new_name}X", x)
      np.save(f"{dir_name}/{new_name}Y", y)



# Extract idle data points from raw data
# psize is the number of samples of one data point (for one electrode)
def idle_points(values, events, psize = 750):
  
  # last idle sample
  idle_size = events[6, 0]
  
  # number of data points
  num_idle_samples = idle_size // psize
  
  # number of samples of all idle points
  idle_size = num_idle_samples * psize

  # x and y arrays of size 288(actions) + number of idle samples
  x = np.zeros((num_idle_samples, psize, 25))
  y = np.zeros(num_idle_samples)
  

  index = 0
  for i in range(0, idle_size, psize):
    x[index] = values[i:i + psize]
    index += 1
  
  x = np.swapaxes(x, 1, 2)

  return x, y


def action_points(values, codes, events, psize = 750):
  
  # convert annotations to mne codes
  cfilter = np.asarray(['769', '770', '771', '772'])
  lis = np.asarray([codes[i] for i in cfilter])

  # filter for classes
  ev = events[np.in1d(events[:, 2], lis)]

  # extract the samples from events
  x = np.zeros((288, psize, 25))
  y = np.zeros(288)

  
  for point in range(len(ev)):
    x[point] = values[ev[point][0]:ev[point][0]+psize]
    y[point] = ev[point][2] - lis[0] + 1
    
  x = np.swapaxes(x, 1, 2)

  return x, y
  


  



