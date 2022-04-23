import numpy as np
import pandas as pd
import os
import mne
import re

def convert_file(filename):
  
  # load the gdf file
  data = mne.io.read_raw_gdf(filename)

  # get data in dataframe format
  dataframe = data.to_data_frame()

  # Get the events 
  events = mne.events_from_annotations(data)
  codes = events[1]
  events = events[0]
  print(events, codes)

  # convert annotations to mne codes
  filter = np.asarray(['769', '770', '771', '772'])
  lis = np.asarray([codes[i] for i in filter])

  # filter for classes
  ev = events[np.in1d(events[:, 2], lis)]

  # extract the samples from events
  x = np.zeros((288, 313, 26))
  y = np.zeros(288)
  
  for point in range(288):
    x[point] = dataframe.values[ev[point][0]:ev[point][0]+313]
    y[point] = ev[point][2] - 6
    
  # Create directory for numpy data
  if not os.path.exists("numpy_data"):
    os.mkdir(os.path.join(os.getcwd(), "numpy_data"))

  # get file name without extension or path
  new_name = os.path.splitext(os.path.basename(filename))[0]
  
  # Save data to numpy arrays
  np.save(f"numpy_data/{new_name}X", x)
  np.save(f"numpy_data/{new_name}Y", y)


def convert_data():
  if os.path.exists("data"):
    datafiles = os.listdir("data")
    
  else:
    print("data directory doesn't exist")
    exit(1)

  for file in datafiles:
    if re.match(r"A0[0-9]T.gdf", file):
      convert_file("data/" +file)
