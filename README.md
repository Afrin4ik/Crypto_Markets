# Crypto Markets

Веб-сервис для мониторинга цены Bitcoin в реальном времени и получения прогноза направления движения цены на ближайший промежуток времени.

Проект разработан как продуктовая часть курсовой работы на тему «Разработка и исследование алгоритмов для прогнозирования финансовых рынков на основе адаптации современных научных методов».

## Кратко о проекте

Сервис представляет собой веб-дашборд для пары `BTCUSDT`. Он получает рыночные данные с биржи Bybit, отображает текущую цену и свечной график, а также позволяет запросить прогноз направления цены Bitcoin с помощью предварительно обученной модели Permutation Decision Tree.

Прогноз является бинарным:

- `UP` - ожидается рост цены;
- `DOWN` - ожидается снижение или отсутствие роста цены.

Модель возвращает направление, уверенность и горизонт прогноза. По умолчанию используется сохранённый артефакт обученной модели из директории `models/`.

## Возможности веб-сервиса

Для пользователя:

- отображение текущей цены Bitcoin в реальном времени;
- отображение 24-часовых рыночных показателей: изменение, максимум, минимум, объём и оборот;
- свечной график BTC/USDT на frontend canvas;
- кнопка «Получить прогноз» для запроса направления движения цены;
- отображение результата прогноза, уверенности и горизонта прогноза;
- статус подключения к потоку рыночных данных.

Для разработчика:

- FastAPI backend с HTTP API и WebSocket;
- получение исторических и real-time данных Bybit через `pybit`;
- общий feature engineering для обучения и inference;
- собственная реализация Permutation Decision Tree;
- CLI-скрипт для повторного обучения модели;
- набор unit-тестов для API, потоков данных, feature engineering, PDT и prediction service.

## Стек и технологии

- Python 3.10+
- FastAPI
- Uvicorn
- Pydantic
- pybit
- NumPy
- Pandas
- scikit-learn
- Optuna
- Joblib
- Pytest
- HTML, CSS, JavaScript

## Архитектура

```text
.
├── app/                                      # FastAPI-приложение и frontend
│   ├── __init__.py
│   ├── main.py                              # создание приложения, lifespan, static files
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                        # HTTP endpoints и WebSocket /ws/market
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                        # настройки и переменные окружения
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── market.py                        # Pydantic-схемы market/prediction данных
│   ├── services/
│   │   ├── __init__.py
│   │   ├── bybit_stream.py                  # исторические и real-time данные Bybit
│   │   ├── market_data.py                   # in-memory состояние рынка и события
│   │   └── prediction.py                    # загрузка PDT-модели и прогноз
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css                   # стили веб-дашборда
│   │   └── js/
│   │       └── app.js                       # WebSocket, canvas-график, запрос прогноза
│   └── templates/
│       └── index.html                       # HTML-страница дашборда
├── ml/                                      # ML-код и обучение модели
│   ├── __init__.py
│   ├── features.py                          # OHLCV-признаки для train/inference
│   ├── train_pdt.py                         # обучение и сохранение PDT-модели
│   └── pdt/
│       ├── __init__.py
│       ├── etc.py                           # расчёт ETC-метрики
│       ├── model.py                         # sklearn-like wrapper PDT-классификатора
│       └── tree.py                          # построение дерева и split logic
├── data/
│   └── Binance_BTCUSDT_2026_minute.csv      # исторические минутные свечи для обучения
├── models/
│   ├── pdt_btc_direction.joblib             # сохранённый артефакт модели
│   └── pdt_btc_direction_metadata.json      # метаданные обучения и качества модели
├── tests/
│   ├── test_api_routes.py                   # тесты HTTP/WebSocket роутов
│   ├── test_bybit_stream.py                 # тесты парсинга сообщений Bybit
│   ├── test_config.py                       # тесты конфигурации
│   ├── test_etc.py                          # тесты ETC
│   ├── test_features.py                     # тесты feature engineering
│   ├── test_market_data.py                  # тесты market data hub
│   ├── test_pdt_model.py                    # тесты PDT-модели
│   ├── test_prediction.py                   # тесты prediction service
│   └── test_prediction_schema.py            # тесты схемы ответа прогноза
├── README.md                                # описание проекта
├── main.py                                  # альтернативная локальная точка запуска
└── requirements.txt                         # Python-зависимости проекта
```

Основной поток работы приложения:

1. При старте backend загружает последние свечи BTC/USDT через HTTP API Bybit.
2. Затем подключается к Bybit WebSocket и обновляет свечи и ticker в реальном времени.
3. Frontend получает начальный снимок рынка через HTTP и дальнейшие обновления через WebSocket `/ws/market`.
4. При нажатии на кнопку «Получить прогноз» frontend отправляет запрос на `/api/predict`.
5. Backend строит признаки из последних свечей, загружает PDT-модель и возвращает прогноз.

## HTTP API

- `GET /` - главная страница дашборда.
- `GET /api/health` - состояние сервиса и подключения к рыночному потоку.
- `GET /api/market/snapshot` - текущий снимок рынка: свечи, ticker, статус.
- `POST /api/predict` - прогноз направления цены Bitcoin.
- `WS /ws/market` - поток обновлений для frontend.

## Быстрый запуск

Создайте виртуальное окружение и установите зависимости:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Запустите веб-сервис:

```bash
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Откройте в браузере:

```text
http://127.0.0.1:8000
```

Альтернативный запуск:

```bash
./.venv/bin/python main.py
```

## Обучение модели

В репозитории уже есть сохранённый артефакт модели:

```text
models/pdt_btc_direction.joblib
models/pdt_btc_direction_metadata.json
```

Если нужно повторно обучить модель, используйте команду:

```bash
./.venv/bin/python -m ml.train_pdt
```

Для обучения используется CSV-файл:

```text
data/Binance_BTCUSDT_2026_minute.csv
```

Скрипт обучения:

- строит OHLCV-признаки из `ml/features.py`;
- формирует бинарную целевую переменную `UP`/`DOWN`;
- использует temporal split `70/15/15`;
- подбирает горизонт прогноза и гиперпараметры через Optuna;
- сохраняет модель и метаданные в директорию `models/`.

После повторного обучения перезапустите FastAPI-сервер, потому что prediction service лениво загружает артефакт модели и держит его в памяти.

## Тесты

Запуск всех тестов:

```bash
./.venv/bin/python -m pytest
```

Тесты покрывают:

- Pydantic-схемы;
- конфигурацию приложения;
- market data hub;
- парсинг сообщений Bybit;
- HTTP/WebSocket роуты;
- feature engineering;
- Permutation Decision Tree;
- prediction service.

## Конфигурация

Параметры можно переопределить через переменные окружения.

| Переменная | Значение по умолчанию | Назначение |
| --- | --- | --- |
| `BYBIT_SYMBOL` | `BTCUSDT` | Торговая пара |
| `BYBIT_CATEGORY` | `spot` | Категория Bybit HTTP API |
| `BYBIT_CHANNEL_TYPE` | `spot` | Тип WebSocket-канала Bybit |
| `BYBIT_TESTNET` | `false` | Использовать Bybit testnet |
| `BYBIT_KLINE_INTERVAL` | `1` | Интервал свечей (в минутах) |
| `BYBIT_HISTORY_LIMIT` | `180` | Количество свечей в памяти сервиса |
| `PDT_MODEL_PATH` | `models/pdt_btc_direction.joblib` | Путь к артефакту модели |
| `PDT_METADATA_PATH` | `models/pdt_btc_direction_metadata.json` | Путь к метаданным модели |
| `PDT_HORIZON_MINUTES` | `10` | Fallback-горизонт до загрузки метаданных |
| `PDT_MAX_ROLLING_WINDOW` | `60` | Максимальное rolling-окно признаков |
| `PDT_MIN_CANDLES` | `80` | Минимум свечей для построения признаков |

## Важные замечания

- Прогноз не является финансовой рекомендацией и используется только в рамках учебного проекта.
- Сервис использует публичные рыночные данные Bybit, поэтому API-ключи не требуются.
- Для стабильной работы прогноза нужно достаточное количество последних свечей. Если данных недостаточно, endpoint `/api/predict` вернёт понятное сообщение о недоступности прогноза.
