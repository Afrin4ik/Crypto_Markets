from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.schemas.market import MarketSnapshot, PredictionResponse, StreamStatus


router = APIRouter()


@router.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    settings = get_settings()
    return FileResponse(Path(settings.templates_dir) / "index.html")


@router.get("/api/health")
async def health(request: Request) -> dict[str, str | bool]:
    status: StreamStatus = request.app.state.market_hub.snapshot().status
    return {"ok": True, "market_stream_connected": status.connected, "message": status.message}


@router.get("/api/market/snapshot", response_model=MarketSnapshot)
async def market_snapshot(request: Request) -> MarketSnapshot:
    return request.app.state.market_hub.snapshot()


@router.post("/api/predict", response_model=PredictionResponse)
async def predict(request: Request) -> PredictionResponse:
    snapshot = request.app.state.market_hub.snapshot()
    try:
        return request.app.state.predictor.predict(snapshot)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.websocket("/ws/market")
async def market_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    market_hub = websocket.app.state.market_hub
    queue = market_hub.subscribe()

    try:
        snapshot = market_hub.snapshot()
        await websocket.send_json({"type": "snapshot", "payload": snapshot.model_dump(mode="json")})
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        market_hub.unsubscribe(queue)
