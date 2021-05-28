import glob
import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_SIZE_GUIDANCE = {
  'compressedHistograms' : 1,
  'images' : 1,
  'scalars' : 0,
  'histograms' : 1
}

def logToPandas(log_path):
  '''
  Convert tensorboard log file to pandas DataFrame
  '''

  try:
    event_acc = EventAccumulator(log_path, DEFAULT_SIZE_GUIDANCE)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']
    log_data = dict()
    for i, tag in enumerate(tags):
      event_list = event_acc.Scalars(tag)
      value = list()
      if i == 0:
        step = list()
      for x in event_list:
        value.append(x.value)
        if i == 0:
          step.append(x.step)
      if len(value) == 25076:
        log_data[tag] = value
      if i == 0:
        log_data['step'] = step
  except:
    print('Error reading event file: {}'.format(log_path))

  data = pd.DataFrame(log_data)
  return data

def logsToPandas(log_paths):
  data = pd.DataFrame()
  for path in log_paths:
    log = logToPadas(path)
    if log is not None:
      if data.shape[0] == 0:
        data = log
      else:
        data = data.append(log, ignore_index=True)

  return data
