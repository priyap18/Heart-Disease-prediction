from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

class ModelTrainer:

    def train_models(self, X_train, X_test, y_train, y_test):

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
        }

        best_model = None
        best_score = 0

        for name, model in models.items():

            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]

            score = roc_auc_score(y_test, y_prob)

            print(name, "ROC-AUC:", round(score, 4))

            if score > best_score:
                best_score = score
                best_model = model

        print("\nBest ROC-AUC:", best_score)

        return best_model