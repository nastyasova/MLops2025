# MLops2025


# Цель проекта

Создать модель, способную автоматически классифицировать отзывы клиентов на позитивные и негативные,  
чтобы сократить ручную обработку отзывов и перераспределить человеческие ресурсы.

## Целевые метрики для продакшена

| Тип метрики | Метрика                                   | Целевое значение | Назначение                 |
|-------------|-------------------------------------------|------------------|----------------------------|
| Business    | Сокращение времени ручной обработки       | ≥ 40 %           | Эффективность работы ОП    |
| Technical   | Среднее время отклика модели              | ≤ 200 мс         | SLA для API                |
| Model       | Accuracy                                  | ≥ 0.90           | Общее качество классификатора |
| Model       | F1 (по негативным отзывам)                | ≥ 0.88           | Выявление проблемных отзывов  |

# Набор данных

Используется открытый датасет [IMDB Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) с сайта [Hugging face](https://huggingface.co/)

Описание с сайта:

Dataset Summary
Large Movie Review Dataset. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well.


**Формат данных:**

- `train`: 25 000 отзывов
- `test`: 25 000 отзывов
- поля:
  - `text` — текст отзыва о фильме
  - `label` — метка тональности:
   - - 0 — негативный отзыв
   - - 1 — позитивный отзыв

## Структура проекта и соответствие заданиям

| Задание | Что реализовано | Где смотреть |
|--------|-----------------|--------------|
| Задание 1 | Версионирование данных и моделей (DVC) | `dvc.yaml`, `dvc.lock`, `data/`, `models/` |
| Задание 2 | Трекинг экспериментов (MLflow) | `src/train.py`, `mlruns/`, `mlflow.db` |
| Задание 3 | Docker-образ для офлайн-инференса | `Dockerfile`, `src/predict.py` |
| Задание 4 | Онлайн-сервис (TorchServe) |  |


# DVC-пайплайн для анализа тональности отзывов


> ⚠️ В рамках учебного задания используется облегчённый режим обучения  
> (subset датасета, `freeze_base=true`), поэтому итоговые метрики ниже продакшен-таргетов.


## Быстрый старт (полный recovery)



```bash
git clone https://github.com/nastyasova/MLops2025.git
cd MLops2025

dvc pull
dvc repro
````

После выполнения команд будут восстановлены: сырой датасет, обработанный датасет, обученная модель, метрики



## Где физически лежат данные и модель

 Не в Git (версионируется DVC)

* `data/raw/imdb`
* `data/processed/imdb_tokenized`
* `models/distilbert-imdb`

 В Git

* код (`src/`, `scripts/`)
* конфигурации (`configs/`)
* `dvc.yaml`, `dvc.lock`
* `.dvc/config`
* `README.md`

Крупные файлы физически хранятся в **DVC remote (S3)**,
Git содержит только метаданные.



Пайплайн описан в `dvc.yaml` и зафиксирован в `dvc.lock`.

Стадии пайплайна

1. prepare — подготовка данных

* Загрузка датасета IMDB
* Токенизация текста

**deps:**

* `scripts/prepare.py`
* `configs/train_config.yaml`

**outs (DVC):**

* `data/raw/imdb`
* `data/processed/imdb_tokenized`


2. train — обучение модели

* Обучение DistilBERT
* Используется subset данных
* Base-модель заморожена (`freeze_base=true`)

**deps:**

* `src/train.py`
* `src/data.py`
* `src/utils.py`
* `configs/train_config.yaml`
* `data/processed/imdb_tokenized`

**outs (DVC):**

* `models/distilbert-imdb`


3. evaluate — оценка модели

* Загрузка сохранённой модели
* Подсчёт метрик качества

**deps:**

* `scripts/evaluate.py`
* `configs/train_config.yaml`
* `data/processed/imdb_tokenized`
* `models/distilbert-imdb`

**metrics:**

* `reports/metrics.yaml`
* `reports/eval_metrics.yaml`


## Конфигурация обучения

Файл: `configs/train_config.yaml`

Ключевые параметры:

* `model_name: distilbert-base-uncased`
* `max_length: 128`
* `batch_size: 16`
* `num_epochs: 1`
* `max_train_samples: 2000`
* `max_eval_samples: 1000`
* `freeze_base: true`


## Метрики

Метрики сохраняются в YAML-файлы:

* `reports/metrics.yaml`
* `reports/eval_metrics.yaml`


Метрики воспроизводимы и пересчитываются командой:

```bash
dvc repro
```


## DVC remote

Используется удалённое хранилище (S3):

```bash
dvc remote list
# yandex  s3://mlops2025-dvc (default)
```

Отправка данных:

```bash
dvc push
```

Проверка синхронизации:

```bash
dvc status -c
```


## Переключение версий данных и модели

Каждый Git-коммит связан с версией данных и модели через `dvc.lock`.

```bash
git checkout <commit_hash>
dvc pull
```

Рабочая директория будет восстановлена в состояние выбранного коммита.


# MLflow — трекинг экспериментов

В проект интегрирован **MLflow** для трекинга экспериментов обучения модели.

Каждый запуск обучения создаёт **отдельный MLflow run**, в котором сохраняются параметры, метрики и артефакты, что обеспечивает полную воспроизводимость экспериментов.


Каждый запуск команды:

```bash
python -m src.train
````

создаёт отдельный эксперимент в MLflow со следующей информацией:

### Параметры (Parameters)

* все гиперпараметры обучения
* параметры из `configs/train_config.yaml`
* batch size, learning rate, число эпох
* параметры выборки (`max_train_samples`, `max_eval_samples`)
* флаг `freeze_base`

### Метрики (Metrics)

* `loss`
* `accuracy`
* `f1`
* метрики валидации и тестирования


### Артефакты (Artifacts)

* обученная модель и токенизатор
* YAML-файлы с метриками:

  * `reports/metrics.yaml`
  * `reports/eval_metrics.yaml`
* файлы описания пайплайна:

  * `dvc.yaml`
  * `dvc.lock`


Для обеспечения связи эксперимента с кодом и данными используются теги:

* `git_commit` — хеш Git-коммита
* `dvc/<stage>/<path>` — DVC-хеши выходных артефактов пайплайна

Таким образом, каждый MLflow run однозначно связан с:

* версией кода
* версией данных
* версией модели


Используется **локальное MLflow-хранилище** (SQLite).

Запуск интерфейса:

```bash
mlflow ui
```

После запуска UI доступен по адресу:

```
http://127.0.0.1:5000
```


Для удобного анализа всех экспериментов реализован скрипт экспорта:

```bash
python scripts/export_mlflow.py
```

Результаты сохраняются в файл:

```
reports/mlflow_runs.csv
```

Файл содержит агрегированную информацию по всем экспериментам:

* run_id
* параметры
* метрики
* время запуска


## Особенности запуска на Windows

Проект разрабатывался и тестировался в среде **Windows (PowerShell)**.

Особенности окружения:

* используется виртуальное окружение `.venv-ml`
* MLflow UI запускается локально (`localhost`)
* используется локальное SQLite-хранилище (`mlflow.db`)
* DVC работает с удалённым S3-хранилищем без Docker

Все команды, приведённые в README, протестированы в Windows-окружении.


## Docker — офлайн-инференс модели

Проект содержит Docker-образ для воспроизводимого **офлайн-инференса** обученной модели.

Контейнер запускает скрипт `src/predict.py`, который:

* загружает сохранённую модель с диска,
* читает входные данные из CSV-файла,
* выполняет предсказание,
* сохраняет результаты в CSV-файл.


### Сборка Docker-образа

```bash
docker build -t ml-app:v1 .
```

Образ собирается на базе `python:3.11-slim` и содержит все необходимые зависимости для инференса модели.



### Запуск инференса

Пример запуска контейнера с монтированием локальной папки с данными:

```bash
docker run --rm -v "${PWD}\sample_data:/data" ml-app:v1 \
  --input_path /data/input.csv \
  --output_path /data/preds.csv
```

После выполнения команды файл с предсказаниями будет сохранён в локальной директории:

```text
sample_data/preds.csv
```


### Формат входных данных

**Вход:** CSV-файл с текстами отзывов.

Обязательное поле:

* `text` — текст отзыва

Пример `input.csv`:

```csv
text
I loved this movie; it was amazing!
Terrible film. Waste of time.
```

Имя колонки можно изменить параметром `--text_col`.



### Формат выходных данных

**Выход:** CSV-файл со следующими полями:

* `text` — исходный текст
* `pred_label` — предсказанный класс (0 — негатив, 1 — позитив)
* `pred_proba_pos` — вероятность позитивного класса

Пример `preds.csv`:

```csv
text,pred_label,pred_proba_pos
I loved this movie; it was amazing!,0,0.13
Terrible film. Waste of time.,0,0.11
```


### Параметры скрипта predict.py

```bash
python -m src.predict \
  --input_path <path_to_input_csv> \
  --output_path <path_to_output_csv> \
  [--model_dir models/distilbert-imdb] \
  [--text_col text] \
  [--batch_size 32]
```


Для уменьшения размера образа используются исключения в `.dockerignore`,
в том числе локальные данные и артефакты (`*.csv`, `mlruns`, `sample_data`).




