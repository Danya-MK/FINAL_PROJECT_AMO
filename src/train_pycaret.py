import os
import pandas as pd
import mlflow
import mlflow.sklearn

from pycaret.classification import setup, compare_models, finalize_model, pull


def train_and_log(train_path: str, current_path: str | None = None, run_name: str = "final_model_package") -> dict:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "wine_quality_automl_retrain")
    mlflow.set_experiment(exp_name)

    train_df = pd.read_csv(train_path)

    if current_path:
        current_df = pd.read_csv(current_path)
        df = pd.concat([train_df, current_df], ignore_index=True)
    else:
        df = train_df

    setup(
        data=df,
        target="target",
        session_id=42,
        fold=5,
        verbose=False,
        log_experiment=True,
        experiment_name=exp_name,
    )

    best = compare_models(sort="AUC")
    final = finalize_model(best)

    # закрываем все активные MLflow run'ы, которые мог оставить PyCaret
    while mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name) as run:
        leaderboard = pull()
        leaderboard_path = "/tmp/pycaret_leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(leaderboard_path, artifact_path="reports")
        mlflow.sklearn.log_model(final, artifact_path="model")
        return {"run_id": run.info.run_id, "experiment_name": exp_name}

    train_df = pd.read_csv(train_path)

    if current_path:
        current_df = pd.read_csv(current_path)
        df = pd.concat([train_df, current_df], ignore_index=True)
    else:
        df = train_df

    setup(
        data=df,
        target="target",
        session_id=42,
        fold=5,
        verbose=False,
        log_experiment=True,
        experiment_name=exp_name,
    )

    best = compare_models(sort="AUC")
    final = finalize_model(best)

    # закрываем все активные MLflow run'ы, которые мог оставить PyCaret
    while mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name) as run:
        leaderboard = pull()
        leaderboard_path = "/tmp/pycaret_leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(leaderboard_path, artifact_path="reports")
        mlflow.sklearn.log_model(final, artifact_path="model")
        return {"run_id": run.info.run_id, "experiment_name": exp_name}