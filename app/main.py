from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.services.bybit_stream import BybitMarketStream
from app.services.market_data import MarketDataHub
from app.services.prediction import PermutationDecisionTreePredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    market_hub = MarketDataHub(settings=settings)
    market_stream = BybitMarketStream(settings=settings, market_hub=market_hub)
    predictor = PermutationDecisionTreePredictor()

    app.state.settings = settings
    app.state.market_hub = market_hub
    app.state.market_stream = market_stream
    app.state.predictor = predictor

    await market_stream.start()
    try:
        yield
    finally:
        await market_stream.stop()


def create_app() -> FastAPI:
    settings: Settings = get_settings()
    fastapi_app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    fastapi_app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    fastapi_app.include_router(router)
    return fastapi_app


app = create_app()
