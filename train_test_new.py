import os
import sys

import warnings
warnings.filterwarnings('ignore')
_ROOT = os.getcwd()
sys.path.append(_ROOT+"/scripts")
sys.path = [os.path.join(_ROOT, "/scripts")] + sys.path

from desired_outputs import desired_outputs
from importlib import reload
from codecarbon import EmissionsTracker

# This is necessary for rllib to get the correct path!
os.chdir(_ROOT+"/scripts")
import train_with_rllib as cpu_trainer

cpu_trainer = reload(cpu_trainer)

cpu_trainer_on, cpu_nego_on_ts = cpu_trainer.trainer(negotiation_on=1, # with naive negotiation
    num_envs=1, 
    train_batch_size=1024, 
    num_episodes=300, 
    lr=0.0005, 
    model_params_save_freq=5000, 
    desired_outputs=desired_outputs, # a list of values that the simulator will output
    num_workers=48)

# Save the results
import pickle
with open("cpu_nego_on_ts_dynamic_group.pkl", "wb") as f:
    pickle.dump(cpu_nego_on_ts, f)