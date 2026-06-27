from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class HyperParameterTuning:

    def tune_logistic_regression(self, X_train, y_train):

        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear", "lbfgs"],
            "penalty": ["l2"]
        }

        grid = GridSearchCV(
            LogisticRegression(max_iter=5000),
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc"
        )

        grid.fit(X_train, y_train)

        print("\nBest Parameters:", grid.best_params_)
        print("Best ROC-AUC:", grid.best_score_)

        return grid.best_estimator_