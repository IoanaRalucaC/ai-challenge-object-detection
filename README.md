# AI Challenge: Real-Time Object Detection App

## Scope

This project is a full-stack application for running object detection on uploaded images.

- The backend exposes an API that performs detection with a YOLO model.
- The frontend provides a web UI for uploading images and viewing detection results.
- Together, they form a local development setup for experimenting with AI-powered visual detection.

## Project Structure

- `backend/`
  - FastAPI service and detection logic
  - Main files include `main.py`, `detect.py`, and model/assets
- `frontend/`
  - Next.js application UI
  - Source code under `src/` (`app`, `components`, `lib`, `types`)
- `install.md`
  - Step-by-step setup and run instructions

## Technologies Used

### Backend

- Python
- FastAPI
- Uvicorn
- Ultralytics YOLO (model inference)

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

## How to Run

For complete installation and run instructions, see:

- [install.md](./install.md)
