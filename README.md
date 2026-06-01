# Crypto Markets

 Репозиторий для курсового проекта на тему «Разработка и исследование алгоритмов для прогнозирования финансовых рынков на основе адаптации современных научных методов».

Продукт: веб-дашборд для real-time мониторинга цены Bitcoin и прогноза на основе Permutation Decision Trees.

## Что реализовано

- FastAPI backend с HTTP API и WebSocket `/ws/market`.
- Получение рыночных данных BTCUSDT с Bybit через официальный `pybit`.
- Первичная загрузка исторических 1m-свечей через HTTP API Bybit.
- Real-time поток свечей и ticker-цены через Bybit WebSocket.
- Frontend на HTML/CSS/JavaScript с canvas-свечным графиком.
- ML-код для бинарного прогноза направления Bitcoin через Permutation Decision Tree.
- Endpoint `/api/predict`, который загружает обученную PDT-модель и возвращает только направление `UP`/`DOWN`, confidence и горизонт прогноза.

## Запуск

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m ml.train_pdt
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

После запуска перейдите по адресу:

```text
http://127.0.0.1:8000
```

Альтернативный запуск:

```bash
./.venv/bin/python main.py
```

## Конфигурация

Параметры можно переопределить через переменные окружения:

- `BYBIT_SYMBOL`, по умолчанию `BTCUSDT`
- `BYBIT_CATEGORY`, по умолчанию `spot`
- `BYBIT_CHANNEL_TYPE`, по умолчанию `spot`
- `BYBIT_TESTNET`, по умолчанию `false`
- `BYBIT_KLINE_INTERVAL`, по умолчанию `1`
- `BYBIT_HISTORY_LIMIT`, по умолчанию `180`
- `PDT_MODEL_PATH`, по умолчанию `models/pdt_btc_direction.joblib`
- `PDT_METADATA_PATH`, по умолчанию `models/pdt_btc_direction_metadata.json`
- `PDT_HORIZON_MINUTES`, по умолчанию `10`
- `PDT_MAX_ROLLING_WINDOW`, по умолчанию `50`
- `PDT_MIN_CANDLES`, по умолчанию `60`

## Обучение PDT

Модель предсказывает направление движения Bitcoin через `PDT_HORIZON_MINUTES` минут (по умолчанию `10`):

- `UP`: `close[t + horizon] > close[t]`
- `DOWN`: `close[t + horizon] <= close[t]`

Исторический CSV лежит в проекте:

```text
data/Binance_BTCUSDT_2026_minute.csv
```

Основная команда обучения:

```bash
python -m ml.train_pdt
```

По умолчанию используется последний хвост из `120_000` минутных свечей, temporal split `70/15/15`, `horizon=10`, небольшой validation grid и PDT с ETC-based split scoring. Артефакты сохраняются сюда:

```text
models/pdt_btc_direction.joblib
models/pdt_btc_direction_metadata.json
```

Быстрая smoke-проверка обучения:

```bash
python -m ml.train_pdt --train-tail-limit 5000 --no-grid \
  --model-path models/pdt_btc_direction_smoke.joblib \
  --metadata-path models/pdt_btc_direction_smoke_metadata.json
```

Используемые признаки строятся только из OHLCV: returns `1/3/5/10`, тело и диапазон свечи, верхняя/нижняя тени, SMA/EMA `10/20/50`, отношения к moving average, RSI 14, rolling volatility `10/30`, volume change и volume rolling mean ratio. Один и тот же код `ml/features.py` используется при обучении и inference.

Фактические метрики последнего полного bounded-обучения на `120_000` строках:

```text
selected_params: max_depth=3, min_samples_leaf=250, max_thresholds=32, etc_sample_limit=256
validation balanced accuracy: 0.5063
test accuracy: 0.5007
test balanced accuracy: 0.5029
test precision: 0.4990
test recall: 0.9387
test F1: 0.6517
test confusion matrix [[DOWN->DOWN, DOWN->UP], [UP->DOWN, UP->UP]]:
[[604, 8411], [547, 8379]]
majority baseline accuracy: 0.4975
previous-direction baseline accuracy: 0.4952
```


## Архитектура

```text
app/
  api/              HTTP и WebSocket роуты
  core/             настройки приложения
  schemas/          Pydantic-схемы данных
  services/         Bybit stream, market hub, PDT prediction service
  static/           CSS и JavaScript
  templates/        HTML страницы
ml/
  features.py       Общий feature engineering для train/inference
  pdt/              ETC и Permutation Decision Tree
  train_pdt.py      CLI обучения и сохранения модели
main.py             локальная точка запуска
requirements.txt    Python-зависимости
```

## Проверки

```bash
python -m pytest
```
