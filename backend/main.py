import os
import time
import uuid
from contextlib import asynccontextmanager

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "yolo26n.pt"

MAX_IMAGE_SIZE = 10 * 1024 * 1024      # 10 MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024      # 50 MB

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
}


OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# YOLO MODEL
# =========================================================

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("Loading YOLO model...")

    model = YOLO(MODEL_NAME)

    print("YOLO model loaded successfully.")

    yield

    print("Shutting down application...")


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="YOLO Detection API",
    description="YOLO image and video detection backend",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =========================================================
# STATIC OUTPUT FILES
# =========================================================

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs",
)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# =========================================================
# IMAGE DETECTION
# =========================================================

@app.post("/api/detect/image")
async def detect_image(
    request: Request,
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # 1. Check file type
    # -----------------------------------------------------

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, and PNG images are supported."
        )

    # -----------------------------------------------------
    # 2. Read file
    # -----------------------------------------------------

    file_bytes = await file.read()

    # -----------------------------------------------------
    # 3. Check file size
    # -----------------------------------------------------

    if len(file_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image file is too large. Maximum size is 10 MB."
        )

    # -----------------------------------------------------
    # 4. Convert bytes → OpenCV image
    # -----------------------------------------------------

    image_array = np.frombuffer(
        file_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the image."
        )

    # -----------------------------------------------------
    # 5. Run YOLO
    # -----------------------------------------------------

    start_time = time.perf_counter()

    results = model(
        image,
        verbose=False
    )

    processing_time = time.perf_counter() - start_time

    result = results[0]

    # -----------------------------------------------------
    # 6. Get annotated image
    # -----------------------------------------------------

    annotated_image = result.plot()

    # -----------------------------------------------------
    # 7. Extract detections
    # -----------------------------------------------------

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0].tolist()
            )

            class_name = model.names[class_id]

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 4),
                "box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            })

    # -----------------------------------------------------
    # 8. Save annotated image
    # -----------------------------------------------------

    output_filename = f"{uuid.uuid4()}.jpg"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    cv2.imwrite(
        output_path,
        annotated_image
    )

    # -----------------------------------------------------
    # 9. Create URL
    # -----------------------------------------------------

    image_url = (
        f"{str(request.base_url).rstrip('/')}"
        f"/outputs/{output_filename}"
    )

    # -----------------------------------------------------
    # 10. Return result
    # -----------------------------------------------------

    return {
        "success": True,
        "filename": file.filename,
        "detections": detections,
        "count": len(detections),
        "processing_time": round(processing_time, 4),
        "image_url": image_url,
    }


# =========================================================
# VIDEO DETECTION
# =========================================================

@app.post("/api/detect/video")
async def detect_video(
    request: Request,
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # 1. Check file type
    # -----------------------------------------------------

    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only MP4 videos are supported."
        )

    # -----------------------------------------------------
    # 2. Read uploaded video
    # -----------------------------------------------------

    file_bytes = await file.read()

    # -----------------------------------------------------
    # 3. Check file size
    # -----------------------------------------------------

    if len(file_bytes) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Video file is too large. Maximum size is 50 MB."
        )

    # -----------------------------------------------------
    # 4. Save temporary input video
    # -----------------------------------------------------

    input_filename = f"input_{uuid.uuid4()}.mp4"

    input_path = os.path.join(
        OUTPUT_DIR,
        input_filename
    )

    with open(input_path, "wb") as f:
        f.write(file_bytes)

    # -----------------------------------------------------
    # 5. Open video
    # -----------------------------------------------------

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        os.remove(input_path)

        raise HTTPException(
            status_code=400,
            detail="Unable to read the video."
        )

    # -----------------------------------------------------
    # 6. Get video information
    # -----------------------------------------------------

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    # -----------------------------------------------------
    # 7. Create output video
    # -----------------------------------------------------

    output_filename = f"{uuid.uuid4()}.mp4"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    # -----------------------------------------------------
    # 8. Process video
    # -----------------------------------------------------

    start_time = time.perf_counter()

    counts = {}

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model(
            frame,
            verbose=False
        )

        result = results[0]

        annotated_frame = result.plot()

        writer.write(
            annotated_frame
        )

        # Count detected classes

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                counts[class_name] = (
                    counts.get(class_name, 0) + 1
                )

    processing_time = (
        time.perf_counter() - start_time
    )

    # -----------------------------------------------------
    # 9. Release resources
    # -----------------------------------------------------

    cap.release()
    writer.release()

    # Remove temporary input video

    os.remove(input_path)

    # -----------------------------------------------------
    # 10. Create output URL
    # -----------------------------------------------------

    video_url = (
        f"{str(request.base_url).rstrip('/')}"
        f"/outputs/{output_filename}"
    )

    # -----------------------------------------------------
    # 11. Return result
    # -----------------------------------------------------

    return {
        "success": True,
        "filename": file.filename,
        "counts": counts,
        "processing_time": round(
            processing_time,
            4
        ),
        "video_url": video_url,
    }