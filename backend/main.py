from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()

model = YOLO("yolo11n.pt")


@app.get("/health")
def health():
    return {"status": "ok"}
