import json

import pandas as pd
import numpy as np
from keras.models import load_model
import pickle

class IrisPredictor:
    def __init__(self):
        self.model = None


    def predict_single_record(self, df):
        if self.model is None:
            with open('model.h5', 'rb') as open_file:
                self.model = pickle.load(open_file)

        y_pred = self.model.predict(df)
        print(y_pred[0])
        status = y_pred[0]
        return status
