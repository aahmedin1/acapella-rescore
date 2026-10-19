"""Acapella Rescore web backend.

Serves the single-page UI and runs rescore jobs in the background.
One credit = one completed job. Credits are tracked per browser cookie
(no login yet); new visitors start with 3 free songs.
"""
import ipaddress
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
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


def refund_credit(user_id: str) -> None:
    c = _db()
    c.execute("UPDATE users SET credits=credits+1 WHERE user_id=?",
              (user_id,))
    c.commit()
    c.close()


# ---------------------------------------------------------------- security
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_JOBS_PER_USER = 2
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_url(url: str) -> bool:
    """SSRF guard: only http(s) URLs whose host resolves to a public IP.
    Blocks localhost, private/LAN, link-local, and cloud-metadata IPs."""
    try:
        p = urllib.parse.urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        host = p.hostname
        if not host or host == "localhost":
            return False
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _check_job_id(job_id: str) -> None:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Bad job id.")


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
                # credit was reserved at creation; keep the deduction
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["stage"] = "Done"
            else:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = "The engine failed. Try another song."
                refund_credit(user_id)
    except Exception as e:  # noqa: BLE001
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
        refund_credit(user_id)


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
    if not url and not file:
        raise HTTPException(400, "Give a link or upload a file.")
    if url and not _validate_url(url):
        raise HTTPException(400, "That link is not allowed.")
    if layers not in ("dum,clap", "dum", "none"):
        raise HTTPException(400, "Bad layers value.")
    # reserve one credit up front: prevents the race where N parallel
    # jobs all pass the balance check and all complete on 1 credit
    if not spend_credit(uid):
        raise HTTPException(402, "Out of credits. Buy more to continue.")
    with jobs_lock:
        running = sum(1 for j in jobs.values()
                      if j["status"] == "running"
                      and j.get("user_id") == uid)
    if running >= MAX_JOBS_PER_USER:
        refund_credit(uid)
        raise HTTPException(429, "Too many jobs running. Wait for one to finish.")
    job_id = uuid.uuid4().hex[:12]
    upload_path = None
    if file is not None:
        job_dir = JOBS / job_id
        job_dir.mkdir(exist_ok=True)
        upload_path = str(job_dir / ("input" + Path(file.filename).suffix))
        # size-capped streaming read: never buffer an unbounded upload
        size = 0
        try:
            with open(upload_path, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "File too large (max 100MB).")
                    f.write(chunk)
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            refund_credit(uid)
            raise
    with jobs_lock:
        jobs[job_id] = {"status": "running", "progress": 0,
                        "stage": "Starting", "error": None,
                        "out_file": None, "created": time.time(),
                        "user_id": uid}
    t = threading.Thread(target=_run_job, args=(
        job_id, uid, url, upload_path, layers, dum_db, clap_db),
        daemon=True)
    t.start()
    return {"job_id": job_id, "user_id": uid}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    _check_job_id(job_id)
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
    _check_job_id(job_id)
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
