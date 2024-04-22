import os, sys

import torch
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dotenv import load_dotenv
load_dotenv()

RANDOM_SEED = 42 # life, the universe and everything” is 42!
MODEL_ID = "google/flan-t5-base"
MODEL_DIR = "models"
BASE_MODEL_PATH = MODEL_DIR + "/" + MODEL_ID
FINE_TUNED_MODEL_NAME = "fine_tuned_t5"
FINE_TUNED_MODEL_PATH = MODEL_DIR +"/"+ FINE_TUNED_MODEL_NAME


## Hyperparameters
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 4
SAVE_TOTAL_LIMIT = 2
EVALUATION_STRATEGY = "epoch"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"