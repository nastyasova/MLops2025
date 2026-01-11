# MLops2025

<!-- Задание 1

 
Выберите тему ML-проекта и сформулируйте бизнес-цель.


Определите целевые метрики для продакшена, например:
- Среднее время отклика сервиса ≤ 200 мс
- Доля неуспешных запросов ≤ 1 %
- Использование памяти/CPU — в пределах SLA
- Качество модели: точность ≥ 90 % или RMSE ≤ 0.1

В README опишите:
* Цель проекта
* Набор данных
* План экспериментов -->

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

---

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

---

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

---

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

---

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

---

## Метрики

Метрики сохраняются в YAML-файлы:

* `reports/metrics.yaml`
* `reports/eval_metrics.yaml`


Метрики воспроизводимы и пересчитываются командой:

```bash
dvc repro
```

---

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

---

## Переключение версий данных и модели

Каждый Git-коммит связан с версией данных и модели через `dvc.lock`.

```bash
git checkout <commit_hash>
dvc pull
```

Рабочая директория будет восстановлена в состояние выбранного коммита.

---


