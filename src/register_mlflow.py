import os
import time
import mlflow
from mlflow.tracking import MlflowClient


def register_run_model_to_registry(run_id: str, stage: str = "Staging", archive_existing: bool = False) -> dict:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    model_name = os.getenv("MLFLOW_MODEL_NAME", "wine_quality_binary_model")
    model_uri = f"runs:/{run_id}/model"

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    for _ in range(30):
        info = client.get_model_version(name=model_name, version=mv.version)
        if info.status == "READY":
            break
        time.sleep(1)

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage=stage,
        archive_existing_versions=archive_existing
    )

    return {"model_name": model_name, "version": mv.version, "stage": stage}