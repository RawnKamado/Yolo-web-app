# YOLO Web Application - Backend

The backend provides the REST API for object detection using FastAPI and Ultralytics YOLO.

## Technologies

- Python
- FastAPI
- Uvicorn
- Ultralytics YOLO
- OpenCV
- NumPy
- Docker

## Features

- Image object detection
- Video object detection
- File type and size validation
- Health check
- Error handling
- Structured logging

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/detect/image` | Detect objects in an image |
| POST | `/api/detect/video` | Detect objects in a video |
| GET | `/docs` | Swagger API documentation |

## YOLO Model

The backend uses the YOLO model:

```text
yolo26n.pt
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## Docker

Build the Docker image:

```bash
docker build -t yolo-backend .
```

Run the container:

```bash
docker run -p 8000:8000 yolo-backend
```

## Logging

The backend provides structured logs containing request ID, route, status, latency, media information, model name, and detection count.

Docker and Render logs can be used to monitor and debug the application.