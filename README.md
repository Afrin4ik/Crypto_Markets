# Crypto Markets

 Репозиторий для курсового проекта на тему «Разработка и исследование алгоритмов для прогнозирования финансовых рынков на основе адаптации современных научных методов».

В качестве продукта разработан веб-дашборд, на котором в реальном времени отображается цена Bitcoin и прогноз её направления на основе модели Permutation Decision Tree. Дашборд получает данные с биржи Bybit, отображает их в виде свечного графика и предоставляет бинарный прогноз движения цены (вверх/вниз) с указанием confidence и горизонта прогноза.

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
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8003
```

После запуска перейдите по адресу:

```text
http://127.0.0.1:8003
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
- `PDT_HORIZON_MINUTES`, по умолчанию `10`; используется как fallback до загрузки обученного artifact
- `PDT_MAX_ROLLING_WINDOW`, по умолчанию `60`
- `PDT_MIN_CANDLES`, по умолчанию `80`

## Обучение PDT

Модель предсказывает направление движения Bitcoin:

- `UP`: `close[t + horizon] > close[t]`
- `DOWN`: `close[t + horizon] <= close[t]`

Горизонт `horizon` выбирается во время обучения из фиксированного набора `10/15/20/30/45/60` минут. Выбранное значение сохраняется в model artifact и автоматически используется backend-ом.

Исторический CSV лежит в проекте:

```text
data/Binance_BTCUSDT_2026_minute.csv
```

Основная команда обучения:

```bash
python -m ml.train_pdt
```

Это единственный путь обучения. Скрипт использует последний хвост из `120_000` минутных свечей, temporal split `70/15/15` и Optuna `TPESampler` для подбора горизонта и PDT-гиперпараметров в пределах лимита примерно до 1 часа. Split внутри PDT считается hybrid score: ETC Gain + Gini impurity gain. Это остаётся PDT-моделью, но разделения лучше оптимизированы под бинарную классификацию направления. Артефакты сохраняются сюда:

```text
models/pdt_btc_direction.joblib
models/pdt_btc_direction_metadata.json
```

После переобучения перезапустите FastAPI-сервер, потому что predictor лениво загружает model artifact и держит его в памяти.

Смотрите не только `confidence`, но и `baseline`-метрики в JSON. Для Bitcoin на минутных свечах высокий confidence может быть следствием перекоса в majority-class, а не реальной предсказательной силы. Поэтому Optuna objective штрафует модели, которые выглядят уверенными, но не обгоняют majority/previous-direction baseline.

Используемые признаки строятся только из OHLCV и приведены к более стационарному виду: returns `1/3/5/10/15/30`, тело и диапазон свечи, верхняя/нижняя тени, разности и отношения SMA/EMA `10/20/50`, RSI 14, rolling volatility `10/30/60`, volume change, volume rolling mean ratio `10/30/60`, положение close внутри rolling high/low channel. Один и тот же код `ml/features.py` используется при обучении и inference.

Предыдущая старая модель была обучена слишком узким grid и выбирала очень неглубокое дерево:

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

После изменений training дополнительно выводит и сохраняет `mean_confidence`, `coverage@0.55`, `accuracy_at_0.55/0.60/0.65`, baseline-метрики и историю лучших Optuna trials.


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
