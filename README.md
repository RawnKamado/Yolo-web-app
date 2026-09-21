# YOLO Web Application

A full-stack object detection web application built with YOLO, FastAPI, and React.

## Features

- Image object detection
- Video object detection
- REST API with FastAPI
- React web interface
- Dockerized backend
- Structured logging
- CI/CD with GitHub Actions
- Cloud deployment with Render

## Project Structure

```text
Yolo-web-app/
├── backend/              # FastAPI + YOLO
├── frontend/             # React + Vite
├── .github/workflows/    # CI/CD workflows
└── README.md
```

## Technologies

### Backend
- Python
- FastAPI
- Ultralytics YOLO
- OpenCV
- NumPy

### Frontend
- React
- Vite
- JavaScript
- CSS

### DevOps
- Docker
- GitHub Actions
- GitHub Container Registry
- Render

## Run Locally

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Configure the frontend to connect to the backend:

```env
VITE_API_URL=http://localhost:8000
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Backend health check |
| POST | `/api/detect/image` | Image detection |
| POST | `/api/detect/video` | Video detection |
| GET | `/docs` | Swagger API documentation |

## Docker

Build and run the backend:

```bash
cd backend
docker build -t yolo-backend .
docker run -p 8000:8000 yolo-backend
```

## CI/CD

GitHub Actions is used for:

- Backend validation
- Frontend build
- Docker image build and publishing
- Automated workflows on GitHub

## Deployment

The application is deployed using Render.

### Frontend

Render Static Site:

https://yolo-web-app-1.onrender.com/

### Backend

Render Web Service:

https://yolo-web-app-hgvv.onrender.com/

### Source Code

GitHub Repository:

https://github.com/RawnKamado/Yolo-web-app

## Logging & Monitoring

The backend uses structured logging for request tracking, status, latency, media information, model information, and detection results.

Docker and Render logs are used for debugging and monitoring.

## Author

**Duong Quang Bao Quoc**