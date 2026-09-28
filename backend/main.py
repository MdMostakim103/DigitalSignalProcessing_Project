# backend/main.py
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"   # prevents Windows AppControl from blocking numba's DLL

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pathlib import Path

from routers.audio_routes import router as audio_router

from fastapi.staticfiles import StaticFiles

app = FastAPI()


class CatchAllErrorsMiddleware(BaseHTTPMiddleware):
    """Turns any unhandled exception into a plain JSON 500 instead of a raw
    ASGI exception. This must be registered *before* CORSMiddleware below —
    Starlette wraps middleware in reverse registration order, so whatever is
    added first ends up innermost, closest to the routes. A plain
    `@app.exception_handler(Exception)` does NOT work for this: Starlette
    special-cases a handler registered for the bare Exception class into
    ServerErrorMiddleware, which sits OUTSIDE CORSMiddleware, so CORS
    headers never get attached to what it returns. A middleware placed here
    instead catches the exception itself and returns an ordinary response,
    so CORSMiddleware (wrapping this from outside) sees a normal completed
    call and adds its headers like it would for any other response.

    Without this, any unhandled exception in a route (a bad upload librosa
    can't decode, an edge-case filter design, ...) bypasses CORSMiddleware
    entirely on its way out, and the browser reports a misleading "blocked
    by CORS policy" error instead of the real failure — which is exactly
    what was happening before this was added.
    """
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"detail": str(exc)})


app.add_middleware(CatchAllErrorsMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Ensure the static folder exists
Path("static/uploads").mkdir(parents=True, exist_ok=True)
Path("static/processed").mkdir(parents=True, exist_ok=True)
Path("static/plots").mkdir(parents=True, exist_ok=True)

# NEW: Plug the router into the main app
app.include_router(audio_router)

app.mount("/static",StaticFiles(directory="static"),name="static")

@app.get("/")
def home(): 
    return {"message": "DSP Audio Engine is Running!"}