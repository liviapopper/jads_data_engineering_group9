import json

import pandas as pd
from keras.models import load_model


class IrisPredictor:
    def __init__(self):
        self.model = None



    def predict_single_record(self, df):
        if self.model is None:
            self.model = load_model('model.h5')
        y_pred = self.model.predict(df)
        species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica' }
        print(species[np.argmax(y_pred)])
        status = species[np.argmax(y_pred)]
        return status
