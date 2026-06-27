from src.prediction import Prediction

predictor = Prediction()

sample = [
    63,
    1,
    4,
    145,
    233,
    1,
    2,
    150,
    0,
    2.3,
    3,
    0,
    6
]

result = predictor.predict(sample)

print("Prediction:", result)