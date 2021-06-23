import glob
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_SIZE_GUIDANCE = {
  'compressedHistograms' : 1,
  'images' : 1,
  'scalars' : 0,
  'histograms' : 1
}

def logToArray(log_path, tags=list()):
  '''
  Convert tensorboard log file to pandas DataFrame
  '''

  try:
    event_acc = EventAccumulator(log_path, DEFAULT_SIZE_GUIDANCE)
    event_acc.Reload()
    tb_tags = event_acc.Tags()['scalars']
    data = dict()
    for i, tag in enumerate(tb_tags):
      if tag in tags:
        event_list = event_acc.Scalars(tag)
        value = list()
        for x in event_list:
          value.append(x.value)
        data[tag] = value
  except:
    print('Error reading event file: {}'.format(log_path))

  return data

def logsToArrays(log_paths, tags=list()):
  data = list()
  for path in log_paths:
    print('Reading event file: {}'.format(path))
    log = logToArray(path, tags=tags)
    data.append(log)

  return data
