# content of test_sysexit.py
import os
#import pytest
import pandas as pd

# content of test_class.py
import iris_predictor

class TestIrisPredictor:
    def test_predict_single_record(self):
        with open('testResources/prediction_request.json') as json_file:
            data = pd.read_json(json_file)
        dp = iris_predictor.IrisPredictor()
        status = dp.predict_single_record(data)
        assert bool(status[0]) is not None
        assert bool(status[0]) is False
