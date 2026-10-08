"""Run the VSL workspace: python test.py (legacy entry point kept intact)."""
from __future__ import annotations

import io
import logging
import secrets
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from recognition import BASE_DIR, LiveSession, ModelRegistry, SAMPLE_FPS, hand_tracker, mp_frame_to_126, tracking_image

registry = ModelRegistry()
sessions: dict[str, LiveSession] = {}
session_lock = Lock()
MAX_UPLOAD = 30 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi"}


@asynccontextmanager
async def lifespan(app):
    await run_in_threadpool(registry.load)
    yield
    for session in list(sessions.values()):
        session.close()
    sessions.clear()


app = FastAPI(title="VSL Studio", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=BASE_DIR / "assets"), name="assets")


@app.get("/")
@app.get("/home.html")
def home_page():
    return FileResponse(BASE_DIR / "home" / "home.html")


@app.get("/upload.html")
def upload_page():
    return FileResponse(BASE_DIR / "upload" / "upload.html")


@app.get("/webcam.html")
def webcam_page():
    return FileResponse(BASE_DIR / "webcam" / "webcam.html")


@app.get("/upload")
def old_upload():
    return RedirectResponse("/upload.html")


@app.get("/webcam")
def old_webcam():
    return RedirectResponse("/webcam.html")


@app.get("/api/models")
def models():
    return {"models": registry.metadata(), "pause_seconds": 6, "sample_fps": SAMPLE_FPS}


def validate_mode(mode):
    if mode not in {"letters", "words", "sentences"}:
        raise HTTPException(400, "Choose letters, single words, or sentences.")
    if not registry.available(mode):
        raise HTTPException(503, registry.errors.get(mode, "This model is unavailable."))


async def read_limited(file, limit):
    chunks, size = [], 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"Choose a file smaller than {limit // (1024 * 1024)} MB.")
        chunks.append(chunk)
    if size == 0:
        raise HTTPException(400, "The file is empty. Choose another file.")
    return b"".join(chunks)


def decode_image(content):
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 20_000_000:
                raise HTTPException(413, "This image is too large. Resize it to under 20 megapixels.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1600, 1600))
            return np.asarray(image).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "This file is not a readable image. Use JPG, PNG, or WebP.")


def predict_video(content, suffix):
    # Temporary clips are deleted on success and failure; filenames never become paths.
    with tempfile.TemporaryDirectory(prefix="vsl-") as folder:
        path = Path(folder) / ("clip" + suffix)
        path.write_bytes(content)
        video = cv2.VideoCapture(str(path))
        tracker = None
        try:
            if not video.isOpened():
                raise HTTPException(400, "This video could not be decoded. Try an MP4 encoded with H.264.")
            fps, total = video.get(cv2.CAP_PROP_FPS), video.get(cv2.CAP_PROP_FRAME_COUNT)
            if not np.isfinite(fps) or fps <= 0 or fps > 240:
                raise HTTPException(400, "This video has invalid timing. Export it as a standard MP4.")
            if total > fps * 45:
                raise HTTPException(400, "Use one sign clip of 45 seconds or less.")
            stride = max(1, round(fps / SAMPLE_FPS))
            vectors = []
            tracker = hand_tracker()
            for index in range(int(fps * 45) + 1):
                if not video.grab():
                    break
                if index % stride:
                    continue
                ok, bgr = video.retrieve()
                if not ok:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tracked = tracker.process(tracking_image(rgb))
                if tracked.multi_hand_landmarks:
                    vectors.append(mp_frame_to_126(tracked))
            return registry.sequence_result(vectors)
        finally:
            video.release()
            if tracker:
                tracker.close()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), mode: str = Form("letters")):
    validate_mode(mode)
    suffix = Path(file.filename or "").suffix.lower()
    if mode == "sentences" and suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, "Sentence recognition needs a video of the whole sign. Choose a video, or switch to Letters for a photo.")
    if mode != "sentences" and suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(400, "Choose a JPG, PNG, or WebP photo for this model.")
    try:
        content = await read_limited(file, MAX_UPLOAD)
        if mode == "sentences":
            return await run_in_threadpool(predict_video, content, suffix)
        rgb = await run_in_threadpool(decode_image, content)
        return await run_in_threadpool(registry.image_result, rgb, mode)
    except HTTPException:
        raise
    except Exception:
        logging.exception("Upload inference failed")
        raise HTTPException(500, "The sign could not be processed. Please try another file.")
    finally:
        await file.close()


@app.post("/api/live/session")
def create_session(mode: str = Form("letters")):
    validate_mode(mode)
    now = time.monotonic()
    with session_lock:
        stale = [key for key, session in sessions.items() if now - session.last_seen > 120]
        for key in stale:
            sessions.pop(key).close()
        if len(sessions) >= 8:
            raise HTTPException(429, "The camera service is busy. Try again in a moment.")
        key = secrets.token_urlsafe(24)
        sessions[key] = LiveSession(mode)
    return {"id": key, "pause_seconds": 6}


@app.post("/api/live/{session_id}/frame")
async def live_frame(session_id: str, file: UploadFile = File(...)):
    with session_lock:
        session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "This sign session has ended. Start a new sign.")
    try:
        content = await read_limited(file, 2 * 1024 * 1024)
        rgb = await run_in_threadpool(decode_image, content)
        return await run_in_threadpool(session.process, rgb, registry)
    finally:
        await file.close()


@app.delete("/api/live/{session_id}")
def delete_session(session_id: str):
    with session_lock:
        session = sessions.pop(session_id, None)
    if session:
        session.close()
    return {"closed": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

