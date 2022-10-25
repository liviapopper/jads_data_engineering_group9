# importing Flask and other modules
import json
import os

import pandas as pd
from flask import Flask, request, render_template, jsonify

from iris_predictor import IrisPredictor

# Flask constructor
app = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
@app.route('/checkiris', methods=["GET", "POST"])
def check_iris():
    if request.method == "POST":
        prediction_input = [
            {
                "Sepal.Length": float(request.form.get("Sepal.Length")),  # getting input in HTML form
                "Sepal.Width": float(request.form.get("Sepal.Width")),
                "Petal.Length": float(request.form.get("Petal.Length")),
                "Petal.Width": float(request.form.get("Petal.Width"))
            }
        ]
        print(prediction_input)
        dp = IrisPredictor()
        df = pd.read_json(json.dumps(prediction_input), orient='records')
        status = dp.predict_single_record(df)
        # return the prediction outcome as a json message. 200 is HTTP status code 200, indicating successful completion
        return jsonify({'result': str(status)}), 200

    return render_template(
        "user_form.html")  # this method is called of HTTP method is GET, e.g., when browsing the link


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
