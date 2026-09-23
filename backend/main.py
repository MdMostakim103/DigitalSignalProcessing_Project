# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from routers.audio_routes import router as audio_router

from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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