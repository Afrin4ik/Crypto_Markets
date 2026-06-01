const state = {
  candles: [],
  ticker: null,
  interval: "1",
  symbol: "BTCUSDT",
  connected: false,
};

const elements = {
  canvas: document.querySelector("#candlestickChart"),
  status: document.querySelector("#streamStatus"),
  statusText: document.querySelector("#streamStatusText"),
  lastPrice: document.querySelector("#lastPrice"),
  dailyChange: document.querySelector("#dailyChange"),
  dailyTurnover: document.querySelector("#dailyTurnover"),
  dailyVolume: document.querySelector("#dailyVolume"),
  dailyHigh: document.querySelector("#dailyHigh"),
  dailyLow: document.querySelector("#dailyLow"),
  chartMeta: document.querySelector("#chartMeta"),
  forecastMeta: document.querySelector("#forecastMeta"),
  forecastResult: document.querySelector("#forecastResult"),
  predictButton: document.querySelector("#predictButton"),
};

const ctx = elements.canvas.getContext("2d");
const resizeObserver = new ResizeObserver(() => drawChart());
resizeObserver.observe(elements.canvas);

function formatPrice(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  const percent = Number(value) * 100;
  const sign = percent > 0 ? "+" : "";
  return `${sign}${percent.toFixed(2)}%`;
}

function formatNumber(value, maximumFractionDigits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits,
  }).format(Number(value));
}

function formatAmount(value, unit, maximumFractionDigits = 2) {
  const formatted = formatNumber(value, maximumFractionDigits);
  return formatted === "—" ? "—" : `${formatted} ${unit}`;
}

function formatConfidence(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return `${Math.round(Number(value) * 100)}%`;
}

function formatPriceDelta(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  const sign = Number(value) > 0 ? "+" : Number(value) < 0 ? "-" : "";
  return `${sign}${formatNumber(Math.abs(Number(value)), 2)}$`;
}

function getDailyPriceChange(ticker) {
  const price = Number(ticker?.last_price);
  const pct = Number(ticker?.price_24h_pct);
  if (!price || Number.isNaN(pct)) {
    return null;
  }
  return price - price / (1 + pct);
}

function updateStatus(status) {
  state.connected = Boolean(status.connected);
  elements.status.classList.toggle("is-online", state.connected);
  elements.status.classList.toggle("is-offline", !state.connected);
  elements.statusText.textContent = status.message || "Нет статуса";
}

function applySnapshot(snapshot) {
  state.symbol = snapshot.symbol;
  state.interval = snapshot.interval;
  state.candles = Array.isArray(snapshot.candles) ? snapshot.candles : [];
  state.ticker = snapshot.ticker;
  updateStatus(snapshot.status);
  renderMarketValues();
  drawChart();
}

function applyCandle(candle) {
  const index = state.candles.findIndex((item) => item.start_ms === candle.start_ms);
  if (index >= 0) {
    state.candles[index] = candle;
  } else {
    state.candles.push(candle);
  }
  state.candles.sort((a, b) => a.start_ms - b.start_ms);
  if (state.candles.length > 180) {
    state.candles = state.candles.slice(-180);
  }
  renderMarketValues();
  drawChart();
}

function applyTicker(ticker) {
  state.ticker = ticker;
  mergeTickerIntoCurrentCandle(ticker);
  renderMarketValues();
  drawChart();
}

function mergeTickerIntoCurrentCandle(ticker) {
  const price = Number(ticker.last_price);
  if (!price) {
    return;
  }
  const timestamp = ticker.timestamp_ms || Date.now();
  const intervalMs = Number(state.interval) * 60 * 1000 || 60 * 1000;
  const start = Math.floor(timestamp / intervalMs) * intervalMs;
  const end = start + intervalMs - 1;
  let latest = state.candles[state.candles.length - 1];

  if (!latest || latest.start_ms < start) {
    latest = {
      start_ms: start,
      end_ms: end,
      interval: state.interval,
      open: price,
      high: price,
      low: price,
      close: price,
      volume: 0,
      turnover: 0,
      confirm: false,
      timestamp_ms: timestamp,
    };
    state.candles.push(latest);
  } else if (latest.start_ms === start) {
    latest.close = price;
    latest.high = Math.max(Number(latest.high), price);
    latest.low = Math.min(Number(latest.low), price);
    latest.timestamp_ms = timestamp;
  }
}

function renderMarketValues() {
  const latestCandle = state.candles[state.candles.length - 1];
  const price = state.ticker?.last_price ?? latestCandle?.close;
  elements.lastPrice.textContent = formatPrice(price);
  const percentChange = formatPercent(state.ticker?.price_24h_pct);
  const priceChange = formatPriceDelta(getDailyPriceChange(state.ticker));
  elements.dailyChange.textContent =
    percentChange === "—" || priceChange === "—" ? percentChange : `${percentChange} (${priceChange})`;
  elements.dailyChange.classList.toggle("positive", Number(state.ticker?.price_24h_pct) > 0);
  elements.dailyChange.classList.toggle("negative", Number(state.ticker?.price_24h_pct) < 0);
  elements.dailyTurnover.textContent = formatAmount(state.ticker?.turnover_24h, "USDT", 0);
  elements.dailyVolume.textContent = formatAmount(state.ticker?.volume_24h, "BTC", 4);
  elements.dailyHigh.textContent = formatPrice(state.ticker?.high_price_24h);
  elements.dailyLow.textContent = formatPrice(state.ticker?.low_price_24h);
  elements.chartMeta.textContent = `${state.interval}m · ${state.candles.length} свечей`;
}

function drawChart() {
  const canvas = elements.canvas;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(320, Math.floor(rect.height));
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const candles = state.candles.slice(-90);
  if (!candles.length) {
    ctx.fillStyle = "#626b73";
    ctx.font = "600 15px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Ожидание рыночных данных Bybit", width / 2, height / 2);
    return;
  }

  const margin = { top: 18, right: 78, bottom: 26, left: 10 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const prices = candles.flatMap((candle) => [Number(candle.high), Number(candle.low)]);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const padding = Math.max((maxPrice - minPrice) * 0.12, maxPrice * 0.0008);
  const low = minPrice - padding;
  const high = maxPrice + padding;
  const priceToY = (price) => margin.top + ((high - price) / (high - low)) * chartHeight;
  const step = chartWidth / Math.max(candles.length, 1);
  const candleWidth = Math.max(3, Math.min(12, step * 0.62));

  drawGrid(width, height, margin, chartHeight, low, high, priceToY);

  candles.forEach((candle, index) => {
    const x = margin.left + step * index + step / 2;
    const open = Number(candle.open);
    const close = Number(candle.close);
    const candleHigh = Number(candle.high);
    const candleLow = Number(candle.low);
    const isUp = close >= open;
    const color = isUp ? "#148a5b" : "#c74343";
    const yOpen = priceToY(open);
    const yClose = priceToY(close);
    const yHigh = priceToY(candleHigh);
    const yLow = priceToY(candleLow);
    const bodyTop = Math.min(yOpen, yClose);
    const bodyHeight = Math.max(2, Math.abs(yClose - yOpen));

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, yHigh);
    ctx.lineTo(x, yLow);
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
  });

  const latestPrice = state.ticker?.last_price ?? candles[candles.length - 1].close;
  drawPriceLine(width, margin, priceToY(Number(latestPrice)), Number(latestPrice));
}

function drawGrid(width, height, margin, chartHeight, low, high, priceToY) {
  ctx.strokeStyle = "#e7ecef";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#626b73";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";

  for (let i = 0; i <= 4; i += 1) {
    const price = low + ((high - low) * i) / 4;
    const y = priceToY(price);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    ctx.fillText(formatPrice(price).replace("US$", "$"), width - margin.right + 8, y);
  }

  ctx.strokeStyle = "#dce2e6";
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + chartHeight);
  ctx.lineTo(width - margin.right, margin.top + chartHeight);
  ctx.stroke();

  ctx.fillStyle = "#9aa3aa";
  ctx.textAlign = "right";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(new Date().toLocaleTimeString("ru-RU"), width - margin.right, height - 8);
}

function drawPriceLine(width, margin, y, price) {
  ctx.strokeStyle = "#087d8f";
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(margin.left, y);
  ctx.lineTo(width - margin.right, y);
  ctx.stroke();
  ctx.setLineDash([]);

  const label = formatPrice(price).replace("US$", "$");
  ctx.font = "700 12px Inter, system-ui, sans-serif";
  const textWidth = ctx.measureText(label).width + 14;
  ctx.fillStyle = "#087d8f";
  ctx.fillRect(width - margin.right + 5, y - 13, textWidth, 26);
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(label, width - margin.right + 12, y);
}

async function fetchSnapshot() {
  const response = await fetch("/api/market/snapshot");
  if (!response.ok) {
    throw new Error("Не удалось получить рыночный снимок");
  }
  applySnapshot(await response.json());
}

function connectMarketSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws/market`);

  socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "snapshot") {
      applySnapshot(data.payload);
    }
    if (data.type === "candle") {
      applyCandle(data.payload);
    }
    if (data.type === "ticker") {
      applyTicker(data.payload);
    }
    if (data.type === "status") {
      updateStatus(data.payload);
    }
  });

  socket.addEventListener("close", () => {
    updateStatus({ connected: false, message: "WebSocket дашборда отключён" });
    window.setTimeout(connectMarketSocket, 2500);
  });
}

async function requestPrediction() {
  elements.predictButton.disabled = true;
  elements.forecastResult.classList.remove("is-up", "is-down");
  elements.forecastResult.querySelector("strong").textContent = "Расчёт...";
  elements.forecastResult.querySelector("span").textContent = "Анализ текущего рыночного снимка";

  try {
    const response = await fetch("/api/predict", { method: "POST" });
    const prediction = await response.json();
    if (!response.ok) {
      throw new Error(prediction.detail || "Прогноз временно недоступен");
    }

    if (!prediction.model_ready || !prediction.direction) {
      elements.forecastResult.querySelector("strong").textContent = "Прогноз недоступен";
      elements.forecastResult.querySelector("span").textContent = prediction.message;
      elements.forecastMeta.textContent = `Цена на момент прогноза: ${formatPrice(prediction.current_price)}`;
      return;
    }

    const isUp = prediction.direction === "UP";
    elements.forecastResult.classList.toggle("is-up", isUp);
    elements.forecastResult.classList.toggle("is-down", !isUp);
    elements.forecastResult.querySelector("strong").textContent = prediction.message;
    elements.forecastResult.querySelector("span").textContent =
      `Уверенность: ${formatConfidence(prediction.confidence)} · ` +
      `Горизонт прогноза: ${prediction.horizon_minutes} минут`;
    elements.forecastMeta.textContent =
      `Цена на момент прогноза: ${formatPrice(prediction.current_price)}`;
  } catch (error) {
    elements.forecastResult.querySelector("strong").textContent = "Прогноз недоступен";
    elements.forecastResult.querySelector("span").textContent = error.message;
  } finally {
    elements.predictButton.disabled = false;
  }
}

elements.predictButton.addEventListener("click", requestPrediction);

fetchSnapshot()
  .catch((error) => updateStatus({ connected: false, message: error.message }))
  .finally(connectMarketSocket);
