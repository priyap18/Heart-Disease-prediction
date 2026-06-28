from src.data_loader import DataLoader
from src.model_training import ModelTrainer
from src.hyperparameter_tuning import HyperParameterTuning

from sklearn.model_selection import train_test_split
import joblib
import os

try:
    # ==========================================
    # STEP 1: Load Dataset
    # ==========================================
    loader = DataLoader("data/Heart_Disease_Prediction.csv")
    df = loader.load_data()

    print("=" * 50)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 50)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nDataset Information:")
    df.info()

    # ==========================================
    # STEP 2: Check Null Values
    # ==========================================
    print("\nNull Values:")
    print(df.isnull().sum())

    if df.isnull().values.any():
        print("\nDataset contains null values.")
    else:
        print("\nNo null values found.")

    # ==========================================
    # STEP 3: Convert Target Column
    # ==========================================
    print("\nUnique Target Values:")
    print(df["Heart Disease"].unique())

    df["Heart Disease"] = df["Heart Disease"].map({
        "Presence": 1,
        "Absence": 0
    })

    # ==========================================
    # STEP 4: Split Features and Target
    # ==========================================
    X = df.drop("Heart Disease", axis=1)
    y = df["Heart Disease"]

    print("\nFeatures Shape:", X.shape)
    print("Target Shape:", y.shape)

    # ==========================================
    # STEP 5: Train-Test Split
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    # ==========================================
    # STEP 6: Train All Models
    # ==========================================
    trainer = ModelTrainer()

    trainer.train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ==========================================
    # STEP 7: Hyperparameter Tuning
    # ==========================================
    tuner = HyperParameterTuning()

    best_model = tuner.tune_logistic_regression(
        X_train,
        y_train
    )

    # ==========================================
    # STEP 8: Save Model
    # ==========================================
    print("\nSaving model...")

    joblib.dump(best_model, "model/model.pkl")

    print("Model saved successfully!")

    # Verify model exists
    if os.path.exists("model/model.pkl"):
        print("model.pkl created successfully.")
    else:
        print("model.pkl was NOT created.")

except Exception as e:
    print("\nAn error occurred:")
    print(e)
