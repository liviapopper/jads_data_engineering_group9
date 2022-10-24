import json

import pandas as pd
from keras.models import load_model
import os
import pickle


class IrisPredictor:
    def __init__(self):
        self.model = None

    def predict_single_record(self, df):
        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        if self.model is None:
            with open('model.h5', 'rb') as open_file:
                self.model = pickle.load(open_file)

        y_pred = self.model.predict(df)
        print(y_pred[0])
        status = (y_pred[0])
        return status

