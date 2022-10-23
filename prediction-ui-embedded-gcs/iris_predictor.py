import json

import pandas as pd
from keras.models import load_model
import os
import numpy as np

from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.models import Model


#import pickle #extra add to read the model

class IrisPredictor:
    def __init__(self):
        self.model = None

    def predict_single_record(self, df):
        if self.model is None:
            self.model = load_model('model.h5')

        y_pred = self.model.predict(df)
        print(y_pred[0])
        status = (y_pred[0])
        return status

