from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("trainduration_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        distance = float(request.form["distance"])
        stops = int(request.form["stops"])

        data = np.array([[distance, stops]])
        result = model.predict(data)[0]

        prediction = round(result, 2)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

