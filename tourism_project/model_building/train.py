"""
Model training step.

Loads the train/test splits produced by prep.py, builds a preprocessing +
XGBoost pipeline, tunes it with GridSearchCV, logs params/metrics to
MLflow, evaluates on the held-out test set, and saves the winning pipeline
into tourism_project/deployment/ so the workflow can commit it to the repo.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, roc_auc_score

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

MODEL_OUTPUT_PATH = "tourism_project/deployment/model.joblib"


def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline():
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )
    model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    return make_pipeline(preprocessor, model)


def main():
    # The workflow starts a local MLflow server before this step; if this
    # is run outside that workflow (e.g. locally), it falls back to a
    # local ./mlruns folder instead of failing.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("tourism-wellness-package")
    except Exception:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("tourism-wellness-package")

    Xtrain, Xtest, ytrain, ytest = load_splits()
    pipeline = build_pipeline()

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    with mlflow.start_run():
        search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=3, n_jobs=-1)
        search.fit(Xtrain, ytrain)

        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)

        preds = best_model.predict(Xtest)
        proba = best_model.predict_proba(Xtest)[:, 1]

        f1 = f1_score(ytest, preds)
        auc = roc_auc_score(ytest, proba)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        print("Best params:", search.best_params_)
        print(classification_report(ytest, preds))
        print(f"F1: {f1:.4f}   ROC-AUC: {auc:.4f}")

        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_OUTPUT_PATH)

        # Logging the model artifact to MLflow is a nice-to-have on top of
        # the params/metrics already logged above; don't let a serializer
        # quirk on a given MLflow version fail the whole training run.
        try:
            mlflow.sklearn.log_model(best_model, "model")
        except Exception as exc:
            print(f"Note: skipped logging model artifact to MLflow ({exc})")

        print(f"Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
