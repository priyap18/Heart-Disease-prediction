import joblib


class Prediction:

    def __init__(self):
        self.model = joblib.load("model/model.pkl")

    def predict(self, features):

        prediction = self.model.predict([features])

        if prediction[0] == 1:
            return "Heart Disease Present"
        else:
            return "No Heart Disease"
