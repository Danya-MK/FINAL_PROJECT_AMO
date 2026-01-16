import os
import sys
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# чтобы Airflow видел src/
DAG_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(DAG_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.drift import compute_psi


def _load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------- INITIAL BASELINE SETUP ----------
def branch_on_initial_baseline(**context):
    """
    Если в MLflow Registry нет Production-версии модели -> делаем initial baseline.
    """
    force = False
    dag_run = context.get("dag_run")
    if dag_run and dag_run.conf:
        force = bool(dag_run.conf.get("force_bootstrap", False))
    if force:
        return "initial_baseline_train"

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    model_name = os.getenv("MLFLOW_MODEL_NAME", "wine_quality_binary_model")

    try:
        latest_prod = client.get_latest_versions(model_name, stages=["Production"])
    except Exception:
        latest_prod = []

    return "initial_baseline_train" if not latest_prod else "skip_initial_baseline"


def initial_baseline_train(**context):
    """
    Обучаем baseline модель только на train.csv и логируем артефакт model/
    Возвращаем run_id, который потом регистрируем в Production.
    """
    from src.train_pycaret import train_and_log

    train_path = os.getenv("TRAIN_DATA_PATH")
    model_info = train_and_log(
        train_path=train_path,
        current_path=None,
        run_name="baseline_model_package"
    )
    context["ti"].xcom_push(key="initial_baseline_model_info", value=model_info)
    return model_info


def initial_baseline_register_prod(**context):
    """
    Регистрируем baseline модель и переводим в Production.
    """
    from src.register_mlflow import register_run_model_to_registry

    model_info = context["ti"].xcom_pull(key="initial_baseline_model_info", task_ids="initial_baseline_train")
    reg = register_run_model_to_registry(
        run_id=model_info["run_id"],
        stage="Production",
        archive_existing=False
    )
    return reg


# ---------- DRIFT + RETRAIN ----------
def check_drift(**context):
    train_path = os.getenv("TRAIN_DATA_PATH")
    curr_path = os.getenv("CURRENT_DATA_PATH")

    threshold = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
    features = [f.strip() for f in os.getenv("DRIFT_FEATURES", "").split(",") if f.strip()]

    train_df = _load_df(train_path)
    curr_df = _load_df(curr_path)

    psi_total, psi_by_feature = compute_psi(train_df, curr_df, features=features, buckets=10)

    report = {
        "psi_total": psi_total,
        "psi_by_feature": psi_by_feature,
        "threshold": threshold,
        "drift": psi_total > threshold
    }
    context["ti"].xcom_push(key="drift_report", value=report)
    return report


def branch_on_drift(**context):
    report = context["ti"].xcom_pull(key="drift_report", task_ids="check_drift")
    return "retrain_automl" if report["drift"] else "no_retrain"


def retrain_automl(**context):
    """
    Retrain на train + current (если drift обнаружен).
    """
    from src.train_pycaret import train_and_log

    train_path = os.getenv("TRAIN_DATA_PATH")
    curr_path = os.getenv("CURRENT_DATA_PATH")

    model_info = train_and_log(
        train_path=train_path,
        current_path=curr_path,
        run_name="final_model_package"
    )
    context["ti"].xcom_push(key="model_info", value=model_info)
    return model_info


def register_model_staging(**context):
    """
    Регистрируем новую модель в Staging.
    """
    from src.register_mlflow import register_run_model_to_registry

    model_info = context["ti"].xcom_pull(key="model_info", task_ids="retrain_automl")
    reg = register_run_model_to_registry(
        run_id=model_info["run_id"],
        stage="Staging",
        archive_existing=False
    )
    return reg


with DAG(
    dag_id="monitoring_retraining_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"owner": "mlops"},
    tags=["mlops", "drift", "pycaret", "mlflow", "retrain"],
) as dag:

    # INITIAL BASELINE SETUP
    t_branch_baseline = BranchPythonOperator(
        task_id="branch_on_initial_baseline",
        python_callable=branch_on_initial_baseline,
    )

    t_skip_baseline = EmptyOperator(task_id="skip_initial_baseline")

    t_baseline_train = PythonOperator(
        task_id="initial_baseline_train",
        python_callable=initial_baseline_train,
    )

    t_baseline_reg_prod = PythonOperator(
        task_id="initial_baseline_register_prod",
        python_callable=initial_baseline_register_prod,
    )

    t_baseline_done = EmptyOperator(
        task_id="initial_baseline_done",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # DRIFT PART
    t_check = PythonOperator(task_id="check_drift", python_callable=check_drift)

    t_branch_drift = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=branch_on_drift,
    )

    t_no = EmptyOperator(task_id="no_retrain")

    t_train = PythonOperator(task_id="retrain_automl", python_callable=retrain_automl)

    t_reg_staging = PythonOperator(task_id="register_model", python_callable=register_model_staging)

    done = EmptyOperator(
        task_id="done",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # wiring
    t_branch_baseline >> t_skip_baseline >> t_baseline_done
    t_branch_baseline >> t_baseline_train >> t_baseline_reg_prod >> t_baseline_done

    t_baseline_done >> t_check >> t_branch_drift
    t_branch_drift >> t_no >> done
    t_branch_drift >> t_train >> t_reg_staging >> done