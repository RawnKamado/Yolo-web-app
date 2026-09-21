# YOLO Web Application - Frontend

The frontend provides the web interface for uploading images and videos and displaying YOLO detection results.

## Technologies

- React
- Vite
- JavaScript
- CSS

## Features

- Image upload
- Video upload
- Object detection results
- Loading state
- Error handling
- Backend API integration

## Configuration

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
```

This variable specifies the URL of the FastAPI backend.

## Run Locally

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

## Production Build

Build the application:

```bash
npm run build
```

The production files are generated in:

```text
dist/
```

## Deployment

The frontend is deployed as a Render Static Site and communicates with the deployed FastAPI backend through the configured `VITE_API_URL`.