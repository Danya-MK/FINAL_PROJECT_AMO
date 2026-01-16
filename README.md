# FINAL_PROJECT: Мониторинг и Retraining ML Модели с Data Drift Detection

## Описание

Этот проект реализует автоматизированный пайплайн машинного обучения для мониторинга дрейфа данных (data drift) и автоматического retraining модели. Проект использует стек технологий: Apache Airflow для оркестрации, MLflow для управления моделями, PyCaret для автоматизированного ML, Docker для контейнеризации и Flask для API обслуживания модели.

Основной функционал:
- Подготовка и обработка датасета Wine Quality
- Тренировка baseline модели
- Мониторинг дрейфа данных в реальном времени
- Автоматический retraining модели при обнаружении дрейфа
- Регистрация и версионирование моделей в MLflow
- API для предсказаний модели

## Структура проекта

```
FINAL_PROJECT/
├── dags/                          # DAG'и Apache Airflow
│   └── monitoring_retraining_dag.py
├── data/                          # Данные
│   ├── train.csv                  # Тренировочные данные
│   ├── current.csv                # Текущие данные для мониторинга
│   └── winequality-red.csv        # Исходный датасет
├── docker/                        # Docker конфигурации
│   ├── airflow/
│   │   ├── Dockerfile
│   │   └── requirements-airflow.txt
│   ├── flask/
│   │   └── Dockerfile
│   ├── mlflow/
│   │   └── Dockerfile
│   └── postgres/
│       └── init.sql
├── logs/                          # Логи Airflow
├── src/                           # Исходный код
│   ├── drift.py                   # Логика обнаружения дрейфа
│   ├── model_serving_api.py       # Flask API для предсказаний
│   ├── register_mlflow.py         # Регистрация моделей в MLflow
│   └── train_pycaret.py           # Тренировка модели с PyCaret
├── docker-compose.yml             # Конфигурация Docker Compose
├── prepare_wine_dataset.py        # Скрипт подготовки данных
└── simulate_data_drift.py         # Скрипт симуляции дрейфа данных
```

## Требования

- Docker и Docker Compose
- Python 3.8+ (для локального запуска)
- Git

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone <URL_репозитория>
cd FINAL_PROJECT
```

### 2. Запуск с помощью Docker Compose

```bash
docker-compose up --build
```

Это запустит следующие сервисы:
- **Airflow**: Оркестратор пайплайна (порт 8080)
- **MLflow**: Сервер для управления моделями (порт 5000)
- **Flask API**: Сервис предсказаний модели (порт 5001)
- **PostgreSQL**: База данных для Airflow и MLflow

### 3. Доступ к интерфейсам

- Airflow UI: http://localhost:8080 (логин: airflow, пароль: airflow)
- MLflow UI: http://localhost:5000
- Model API: http://localhost:5001

## Использование

### Запуск пайплайна

1. Откройте Airflow UI
2. Включите DAG `monitoring_retraining_dag`
3. Запустите DAG вручную или дождитесь scheduled запуска

### DAG задачи

- **initial_baseline_train**: Тренировка начальной модели
- **check_drift**: Проверка дрейфа данных
- **retrain_automl**: Retraining модели при дрейфе
- **register_model**: Регистрация модели в MLflow

### API предсказаний

Отправьте POST запрос на `http://localhost:5001/predict` с JSON данными:

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

### Симуляция дрейфа

Запустите скрипт для симуляции дрейфа данных:

```bash
python simulate_data_drift.py
```

## Конфигурация

Основные настройки в `docker-compose.yml`:
- Переменные окружения для Airflow
- Конфигурация баз данных
- Порты сервисов
