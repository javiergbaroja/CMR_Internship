import numpy as np
import torch
import random
import logging
from datetime import datetime
import os
from natsort import natsorted

import json

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_everything(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

def shuffle_lists(*ls):
  
  l =list(zip(*ls))
  random.shuffle(l)

  return zip(*l)

def logger_setup(level=logging.INFO, folder:str="logs_folder") -> logging.Logger:

    log_file = get_log_file(folder)

    logging.basicConfig(
        filename=log_file,
        level=level, 
        format= '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(level)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)

    logger.info(f"Logging to: {log_file}")

    return logger
    
def get_log_file(folder:str):

    os.makedirs(folder, exist_ok=True)
    date = datetime.now().strftime("%Y_%m_%d")
    log_file = os.path.join(folder, date)
    previous = natsorted([f.path for f in os.scandir(folder) if log_file in f.path and os.path.isfile(f.path)])
    if len(previous)==0:
        log_file += "_0_log.out"
    else:
        log_file = previous[-1].split("_")
        log_file[-2] = str(int(log_file[-2]) + 1)
        log_file = "_".join(log_file)
    return log_file

def get_exp_num(folder:str="Results", cv=False):

    os.makedirs(folder, exist_ok=True)

    date = datetime.now().strftime("%Y_%m_%d")
    path = os.path.join(folder, date)
    previous = natsorted([f.path for f in os.scandir(folder) if path in f.path and os.path.isdir(f.path)])
    
    if len(previous)==0:
        expnum = "0"
    else:
        expnum = previous[-1].split(folder)[1].split("_")[3]
        expnum = str(int(expnum) + 1) if cv == False else str(int(expnum))
    
    return expnum


def parse_json_file(file_path):
    """
    Parses the file specified in path, with some additional security checks.
    :param file_path: File path to parse
    :returns : The contents in JSON format, or None if the file is empty.
    """
    if not file_path or os.stat(file_path).st_size == 0:
        return None

    with open(file_path, "r") as json_file:
        file_contents = json_file.read()

    return parse_json(file_contents)


def _replace_single_quotes(text):
    replaced_text = text.replace("'", '"')

    return replaced_text

def parse_json(text):
    """
    Parses the specified text as JSON, with some additional security checks.
    :param text: Text to parse.
    :returns : The parsed results, or None if the string is empty.
    """
    if not text:
        return None
    else:
        try:
            return json.loads(text)
        except Exception:
            return json.loads(_replace_single_quotes(text))

def save_json(path:str, file_name:str, object):
    os.makedirs(os.path.join(path), exist_ok=True)
    with open(os.path.join(path, file_name), 'w') as f:
        json.dump(object, f)

