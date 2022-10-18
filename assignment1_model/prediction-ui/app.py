# importing Flask and other modules
import json

import requests
from flask import Flask, request, render_template

# Flask constructor
app = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
@app.route('/checkiris', methods=["GET", "POST"])
def check_iris():
    if request.method == "POST":
        prediction_input = [
            {
                "Sepal.Length": int(request.form.get("Sepal.Length")),  # getting input with name = ntp in HTML form
                "Sepal.Width": int(request.form.get("Sepal.Width")),  # getting input with name = pgc in HTML form
                "Petal.Length": int(request.form.get("Petal.Length")),
                "Petal.Width": int(request.form.get("Petal.Width"))
            }
        ]
        print(prediction_input)
        # use requests library to execute the prediction service API by sending a HTTP POST request
        # localhost or 127.0.0.1 is used when the applications are on the same machine.
        res = requests.post('http://localhost:5000/iris_predictor/', json=json.loads(json.dumps(prediction_input)))
        print(res.status_code)
        result = res.json()
        return result
    return render_template(
        "user_form.html")  # this method is called of HTTP method is GET, e.g., when browsing the link


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
