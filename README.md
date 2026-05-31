# Crypto Markets

 Репозиторий для курсового проекта на тему «Разработка и исследование алгоритмов для прогнозирования финансовых рынков на основе адаптации современных научных методов».

Продукт: веб-дашборд для real-time мониторинга цены Bitcoin и прогноза на основе Permutation Decision Trees.

## Что реализовано

- FastAPI backend с HTTP API и WebSocket `/ws/market`.
- Получение рыночных данных BTCUSDT с Bybit через официальный `pybit`.
- Первичная загрузка исторических 1m-свечей через HTTP API Bybit.
- Real-time поток свечей и ticker-цены через Bybit WebSocket.
- Frontend на HTML/CSS/JavaScript с canvas-свечным графиком.
- Endpoint `/api/predict`, подготовленный под будущую PDT-модель. Сейчас он возвращает демонстрационную заглушку.

## Запуск

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
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

## Архитектура

```text
app/
  api/              HTTP и WebSocket роуты
  core/             настройки приложения
  schemas/          Pydantic-схемы данных
  services/         Bybit stream, market hub, predictor stub
  static/           CSS и JavaScript
  templates/        HTML страницы
main.py             локальная точка запуска
requirements.txt    Python-зависимости
```
