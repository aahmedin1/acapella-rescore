"""Acapella Rescore web backend.

Serves the single-page UI and runs rescore jobs in the background.
One credit = one completed job. Credits are tracked per browser cookie
(no login yet); new visitors start with 3 free songs.
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import Cookie, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
JOBS = BASE / "jobs"
DB = BASE / "credits.db"
ENGINE = ROOT / "rescore.py"
PY = sys.executable

FREE_CREDITS = 3
JOBS.mkdir(exist_ok=True)

app = FastAPI(title="Acapella Rescore")

# ---------------------------------------------------------------- credits
def _db():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS users "
              "(user_id TEXT PRIMARY KEY, credits INTEGER)")
    return c


def credits_for(user_id: str) -> int:
    c = _db()
    row = c.execute("SELECT credits FROM users WHERE user_id=?",
                    (user_id,)).fetchone()
    if row is None:
        c.execute("INSERT INTO users VALUES (?, ?)", (user_id, FREE_CREDITS))
        c.commit()
        c.close()
        return FREE_CREDITS
    c.close()
    return row[0]


def spend_credit(user_id: str) -> bool:
    c = _db()
    row = c.execute("SELECT credits FROM users WHERE user_id=?",
                    (user_id,)).fetchone()
    if row is None or row[0] <= 0:
        c.close()
        return False
    c.execute("UPDATE users SET credits=credits-1 WHERE user_id=?",
              (user_id,))
    c.commit()
    c.close()
    return True


# ---------------------------------------------------------------- jobs
# job_id -> dict(status, progress, stage, error, out_file, created)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

STAGES = [
    (r"downloading", 5, "Downloading the song"),
    (r"separating", 20, "Separating voice from music"),
    (r"beat-tracking", 55, "Finding the beat"),
    (r"laying down|lifting groove", 65, "Building the groove"),
    (r"beat level", 75, "Laying down the beats"),
    (r"doubling the lead", 85, "Mixing"),
    (r"wrote ", 100, "Done"),
]


def _run_job(job_id: str, user_id: str, url: str | None,
             upload_path: str | None, layers: str,
             dum_db: float, clap_db: float):
    job_dir = JOBS / job_id
    job_dir.mkdir(exist_ok=True)
    out = job_dir / "output.wav"
    cmd = [PY, str(ENGINE), "--out", str(out),
           "--perc-db", "13", "--backing", "haas-vocal",
           "--kit", "duff", "--groove", "lifted",
           "--layers", layers,
           "--dum-db", str(dum_db), "--clap-db", str(clap_db)]
    if url:
        cmd += ["--url", url]
    else:
        cmd += ["--mix", upload_path]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, cwd=str(ROOT))
        for line in proc.stdout:
            for pat, pct, label in STAGES:
                if re.search(pat, line, re.IGNORECASE):
                    with jobs_lock:
                        jobs[job_id]["progress"] = pct
                        jobs[job_id]["stage"] = label
                    break
        proc.wait()
        with jobs_lock:
            if proc.returncode == 0 and out.exists():
                if spend_credit(user_id):
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["progress"] = 100
                    jobs[job_id]["stage"] = "Done"
                else:
                    jobs[job_id]["status"] = "error"
                    jobs[job_id]["error"] = "Out of credits."
            else:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = "The engine failed. Try another song."
    except Exception as e:  # noqa: BLE001
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)


def _ensure_user(user_id: str | None) -> str:
    return user_id or str(uuid.uuid4())


# ---------------------------------------------------------------- routes
@app.get("/api/credits")
def get_credits(user_id: str | None = Cookie(default=None)):
    uid = _ensure_user(user_id)
    return {"user_id": uid, "credits": credits_for(uid)}


@app.post("/api/jobs")
async def create_job(url: str | None = Form(default=None),
                     layers: str = Form(default="dum,clap"),
                     dum_db: float = Form(default=0.0),
                     clap_db: float = Form(default=0.0),
                     file: UploadFile | None = File(default=None),
                     user_id: str | None = Cookie(default=None)):
    uid = _ensure_user(user_id)
    if credits_for(uid) <= 0:
        raise HTTPException(402, "Out of credits. Buy more to continue.")
    if not url and not file:
        raise HTTPException(400, "Give a link or upload a file.")
    if layers not in ("dum,clap", "dum", "none"):
        raise HTTPException(400, "Bad layers value.")
    job_id = uuid.uuid4().hex[:12]
    upload_path = None
    if file is not None:
        job_dir = JOBS / job_id
        job_dir.mkdir(exist_ok=True)
        upload_path = str(job_dir / ("input" + Path(file.filename).suffix))
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    with jobs_lock:
        jobs[job_id] = {"status": "running", "progress": 0,
                        "stage": "Starting", "error": None,
                        "out_file": None, "created": time.time()}
    t = threading.Thread(target=_run_job, args=(
        job_id, uid, url, upload_path, layers, dum_db, clap_db),
        daemon=True)
    t.start()
    return {"job_id": job_id, "user_id": uid}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        j = jobs.get(job_id)
    if j is None:
        # job may predate this process restart; look for a finished file
        out = JOBS / job_id / "output.wav"
        if out.exists():
            return {"status": "done", "progress": 100, "stage": "Done",
                    "error": None, "has_audio": True}
        raise HTTPException(404, "Unknown job.")
    return {"status": j["status"], "progress": j["progress"],
            "stage": j["stage"], "error": j["error"],
            "has_audio": j["status"] == "done"}


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str):
    out = JOBS / job_id / "output.wav"
    if not out.exists():
        raise HTTPException(404, "Audio not ready.")
    return FileResponse(out, media_type="audio/wav",
                        filename="acapella-rescore.wav")


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "static" / "index.html").read_text()


app.mount("/static", StaticFiles(directory=str(BASE / "static")),
          name="static")
