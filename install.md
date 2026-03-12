# Install and Run Guide

This project has two apps:

- `backend` (FastAPI + YOLO)
- `frontend` (Next.js)

Use the PowerShell commands below from the project root (the folder that contains both `backend` and `frontend`).

## 1) Get the Project on Your Computer

If you do not already have the code on your machine:

```powershell
git clone <your-repo-url>
cd <repo-folder-name>
```

If you already have the code, just open a terminal in your existing project root.

Optional: open the project in Visual Studio Code by clicking on project.code-workspace file

## 2) Prerequisites

Install these first:

- Python 3.11+ (recommended)
- Node.js 20+
- npm (comes with Node.js)

Check versions:

```powershell
python --version
node --version
npm --version
```

## 3) Backend Setup (FastAPI)

Open a terminal in the project root and run:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional: if you want to override defaults, create `backend/.env`:

```env
MODEL_PATH=yolov8m.pt
ALLOWED_ORIGINS=http://localhost:3000
```

Start the backend server:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend URLs:

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

## 4) Frontend Setup (Next.js)

Open a second terminal in the project root and run:

```powershell
cd frontend
npm install
```

Optional: create `frontend/.env.local` if you need a custom backend URL:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Start the frontend dev server:

```powershell
npm run dev
```

Frontend URL:

- App: `http://localhost:3000`

## 5) Daily Start Commands

After first-time setup, start both services with two terminals.

Terminal 1 (backend):

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 (frontend):

```powershell
cd frontend
npm run dev
```
