import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def _resolve_data_path(data_path: str) -> str:
    # Template submission menggunakan default "MLProject/namadataset_preprocessing".
    # Saat `mlflow run MLProject/`, working directory biasanya sudah berada di folder MLProject,
    # sehingga path tersebut tidak selalu ada. Fallback agar tetap jalan.
    if os.path.exists(data_path):
        return data_path

    if data_path.startswith("MLProject/"):
        candidate = data_path.replace("MLProject/", "", 1)
        if os.path.exists(candidate):
            return candidate

    return data_path


def train(data_path: str):
    data_path = _resolve_data_path(data_path)

    X_train = pd.read_csv(f"{data_path}/X_train.csv")
    X_test = pd.read_csv(f"{data_path}/X_test.csv")
    y_train = pd.read_csv(f"{data_path}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{data_path}/y_test.csv").squeeze()

    # IMPORTANT:
    # When executed via `mlflow run MLProject/`, MLflow Projects already creates
    # an active run and sets the experiment. Calling set_experiment/start_run again
    # can cause experiment/run ID conflicts.
    mlflow.sklearn.autolog()

    def _fit_and_log():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1-Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")

    if mlflow.active_run() is None:
        with mlflow.start_run():
            _fit_and_log()
    else:
        _fit_and_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="MLProject/namadataset_preprocessing"
    )
    args = parser.parse_args()
    train(args.data_path)
