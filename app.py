from flask import Flask, render_template, request
from src.prediction import Prediction

app = Flask(__name__)

predictor = Prediction()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["chest_pain_type"]),
        float(request.form["BP"]),
        float(request.form["cholesterol"]),
        float(request.form["FBS_over_120"]),
        float(request.form["EKG_results"]),
        float(request.form["max_HR"]),
        float(request.form["exercise_angina"]),
        float(request.form["ST_depression"]),
        float(request.form["slope_of_ST"]),
        float(request.form["number_of_vessels_fluro"]),
        float(request.form["thallium"])
    ]

    result = predictor.predict(features)

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)