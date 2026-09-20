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

from logging_config import setup_logging, logger


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
# LOGGING
# =========================================================

setup_logging()


# =========================================================
# YOLO MODEL
# =========================================================

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    logger.info(
        "model_loading | model=%s",
        MODEL_NAME,
    )

    try:
        model = YOLO(MODEL_NAME)

        logger.info(
            "model_loaded | model=%s",
            MODEL_NAME,
        )

    except Exception:
        logger.exception(
            "model_load_failed | model=%s",
            MODEL_NAME,
        )
        raise

    yield

    logger.info(
        "application_shutdown | model=%s",
        MODEL_NAME,
    )


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
# REQUEST LOGGING
# =========================================================

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Generate a unique ID for each request
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Start measuring request processing time
    start_time = time.perf_counter()

    response = None

    try:
        response = await call_next(request)
        return response

    except Exception:
        # Log unexpected errors
        logger.exception(
            "request_error | request_id=%s | method=%s | route=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise

    finally:
        # Calculate request latency
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        status_code = (
            response.status_code
            if response is not None
            else 500
        )

        # Log request metadata
        logger.info(
            "request | request_id=%s | method=%s | "
            "route=%s | status=%s | latency_ms=%s",
            request_id,
            request.method,
            request.url.path,
            status_code,
            latency_ms,
        )


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
        "https://yolo-web-app-1.onrender.com",
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
    file: UploadFile = File(...),
):

    # -----------------------------------------------------
    # 1. Get request information
    # -----------------------------------------------------

    request_id = request.state.request_id

    # -----------------------------------------------------
    # 2. Check file type
    # -----------------------------------------------------

    if file.content_type not in ALLOWED_IMAGE_TYPES:

        logger.warning(
            "invalid_image_type | request_id=%s | "
            "media_type=%s",
            request_id,
            file.content_type,
        )

        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, and PNG images are supported.",
        )

    # -----------------------------------------------------
    # 3. Read file
    # -----------------------------------------------------

    file_bytes = await file.read()

    media_size = len(file_bytes)
    media_type = file.content_type

    # -----------------------------------------------------
    # 4. Check file size
    # -----------------------------------------------------

    if media_size > MAX_IMAGE_SIZE:

        logger.warning(
            "image_too_large | request_id=%s | "
            "media_type=%s | media_size=%s",
            request_id,
            media_type,
            media_size,
        )

        raise HTTPException(
            status_code=413,
            detail="Image file is too large. Maximum size is 10 MB.",
        )

    # -----------------------------------------------------
    # 5. Convert bytes → OpenCV image
    # -----------------------------------------------------

    image_array = np.frombuffer(
        file_bytes,
        np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        logger.warning(
            "invalid_image | request_id=%s | "
            "media_type=%s | media_size=%s",
            request_id,
            media_type,
            media_size,
        )

        raise HTTPException(
            status_code=400,
            detail="Unable to read the image.",
        )

    # -----------------------------------------------------
    # 6. Run YOLO
    # -----------------------------------------------------

    start_time = time.perf_counter()

    try:

        results = model.predict(
            source=image,
            imgsz=320,
            conf=0.25,
            device="cpu",
            verbose=False,
        )

    except Exception:

        logger.exception(
            "yolo_image_error | request_id=%s | "
            "media_type=%s | media_size=%s | model=%s",
            request_id,
            media_type,
            media_size,
            MODEL_NAME,
        )

        raise

    processing_time = time.perf_counter() - start_time

    result = results[0]

    # -----------------------------------------------------
    # 7. Get annotated image
    # -----------------------------------------------------

    annotated_image = result.plot()

    # -----------------------------------------------------
    # 8. Extract detections
    # -----------------------------------------------------

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0].tolist(),
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
                },
            })

    # -----------------------------------------------------
    # 9. Log detection result
    # -----------------------------------------------------

    logger.info(
        "detection | request_id=%s | media_type=%s | "
        "media_size=%s | model=%s | detection_count=%s",
        request_id,
        media_type,
        media_size,
        MODEL_NAME,
        len(detections),
    )

    # -----------------------------------------------------
    # 10. Save annotated image
    # -----------------------------------------------------

    output_filename = f"{uuid.uuid4()}.jpg"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename,
    )

    cv2.imwrite(
        output_path,
        annotated_image,
    )

    # -----------------------------------------------------
    # 11. Create URL
    # -----------------------------------------------------

    image_url = (
        f"{str(request.base_url).rstrip('/')}"
        f"/outputs/{output_filename}"
    )

    # -----------------------------------------------------
    # 12. Return result
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
def detect_video(
    request: Request,
    file: UploadFile = File(...),
):

    # -----------------------------------------------------
    # 1. Get request information
    # -----------------------------------------------------

    request_id = request.state.request_id

    # -----------------------------------------------------
    # 2. Check file type
    # -----------------------------------------------------

    if file.content_type not in ALLOWED_VIDEO_TYPES:

        logger.warning(
            "invalid_video_type | request_id=%s | "
            "media_type=%s",
            request_id,
            file.content_type,
        )

        raise HTTPException(
            status_code=400,
            detail="Only MP4 videos are supported.",
        )

    # -----------------------------------------------------
    # 3. Save uploaded video in chunks
    # -----------------------------------------------------

    media_type = file.content_type

    input_filename = f"input_{uuid.uuid4()}.mp4"

    input_path = os.path.join(
        OUTPUT_DIR,
        input_filename,
    )

    media_size = 0

    with open(input_path, "wb") as f:

        while True:

            chunk = file.file.read(1024 * 1024)  # 1 MB

            if not chunk:
                break

            media_size += len(chunk)

            if media_size > MAX_VIDEO_SIZE:

                f.close()

                if os.path.exists(input_path):
                    os.remove(input_path)

                logger.warning(
                    "video_too_large | request_id=%s | "
                    "media_type=%s | media_size=%s",
                    request_id,
                    media_type,
                    media_size,
                )

                raise HTTPException(
                    status_code=413,
                    detail="Video file is too large. Maximum size is 50 MB.",
                )

            f.write(chunk)

    # -----------------------------------------------------
    # 4. Open video
    # -----------------------------------------------------

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():

        if os.path.exists(input_path):
            os.remove(input_path)

        logger.warning(
            "invalid_video | request_id=%s | "
            "media_type=%s | media_size=%s",
            request_id,
            media_type,
            media_size,
        )

        raise HTTPException(
            status_code=400,
            detail="Unable to read the video.",
        )

    # -----------------------------------------------------
    # 5. Get video information
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
    # 6. Create output video
    # -----------------------------------------------------

    output_filename = f"{uuid.uuid4()}.mp4"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        cap.release()

        if os.path.exists(input_path):
            os.remove(input_path)

        logger.error(
            "video_writer_error | request_id=%s | "
            "media_type=%s | media_size=%s",
            request_id,
            media_type,
            media_size,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to create output video.",
        )

    # -----------------------------------------------------
    # 7. Process video
    # -----------------------------------------------------

    start_time = time.perf_counter()

    counts = {}

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            try:

                results = model.predict(
                    source=frame,
                    imgsz=320,
                    conf=0.25,
                    device="cpu",
                    verbose=False,
                )

            except Exception:

                logger.exception(
                    "yolo_video_error | request_id=%s | "
                    "media_type=%s | media_size=%s | model=%s",
                    request_id,
                    media_type,
                    media_size,
                    MODEL_NAME,
                )

                raise

            result = results[0]

            annotated_frame = result.plot()

            writer.write(
                annotated_frame
            )

            # Count detected classes
            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = model.names[
                        class_id
                    ]

                    counts[class_name] = (
                        counts.get(class_name, 0) + 1
                    )

    except Exception:

        cap.release()
        writer.release()

        if os.path.exists(input_path):
            os.remove(input_path)

        if os.path.exists(output_path):
            os.remove(output_path)

        raise

    finally:

        cap.release()
        writer.release()

    # -----------------------------------------------------
    # 8. Calculate processing time
    # -----------------------------------------------------

    processing_time = (
        time.perf_counter() - start_time
    )

    # -----------------------------------------------------
    # 9. Remove temporary input video
    # -----------------------------------------------------

    if os.path.exists(input_path):
        os.remove(input_path)

    # -----------------------------------------------------
    # 10. Calculate total detections
    # -----------------------------------------------------

    total_detections = sum(
        counts.values()
    )

    # -----------------------------------------------------
    # 11. Log video detection result
    # -----------------------------------------------------

    logger.info(
        "detection | request_id=%s | media_type=%s | "
        "media_size=%s | model=%s | detection_count=%s",
        request_id,
        media_type,
        media_size,
        MODEL_NAME,
        total_detections,
    )

    # -----------------------------------------------------
    # 12. Create URL
    # -----------------------------------------------------

    video_url = (
        f"{str(request.base_url).rstrip('/')}"
        f"/outputs/{output_filename}"
    )

    # -----------------------------------------------------
    # 13. Return result
    # -----------------------------------------------------

    return {
        "success": True,
        "filename": file.filename,
        "counts": counts,
        "processing_time": round(
            processing_time,
            4,
        ),
        "video_url": video_url,
    }
