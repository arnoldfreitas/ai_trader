import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras

class env():
    '''
    Environment Class.
    
    Interaction of the agent with the environment. Calculates utility/reward. Observes states and executes actions (env.step()).


    '''