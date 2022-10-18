import json

import pandas as pd
from flask import jsonify
from keras.models import load_model


class IrisPredictor:
    def __init__(self):
        self.model = None

    def predict_single_record(self, prediction_input):
        print(prediction_input)
        if self.model is None:
            self.model = load_model('iris_flower_classification_model_01.h5')
        print(json.dumps(prediction_input))
        df = pd.read_json(json.dumps(prediction_input), orient='records')
        print(df)
        y_pred = self.model.predict(df)
        print(y_pred[0])
        # return the prediction outcome as a json message. 200 is HTTP status code 200, indicating successful completion
        return jsonify({'Flower type result': str(y_pred[0])}), 200
