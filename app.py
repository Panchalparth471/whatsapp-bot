# app.py
"""
Twilio WhatsApp bot backend (Flask)
- Uses MongoDB for sessions, cache, videos, tasks
- Receives inbound messages from Twilio webhook (/whatsapp/webhook)
- Enqueues generation tasks, uses Replicate to generate videos
- Uploads generated videos to Cloudinary and serves them from there
- Sends generated video back using Twilio (media_url must be publicly reachable)
- Exposes /media/<filename> to serve generated files (falls back to Cloudinary)
"""

import os
import time
import uuid
import logging
import threading
import queue
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from flask import Flask, request, send_file, jsonify, redirect
import requests
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# --- Optional SDKs ---
try:
    import cloudinary
    import cloudinary.uploader
except Exception:
    cloudinary = None

try:
    from twilio.rest import Client as TwilioClient
    from twilio.twiml.messaging_response import MessagingResponse
except Exception:
    TwilioClient = None
    MessagingResponse = None

try:
    from pymongo import MongoClient, ASCENDING
except Exception:
    raise RuntimeError("pymongo required. Install with: pip install pymongo")

try:
    import replicate
except Exception:
    replicate = None

try:
    import openai
except Exception:
    openai = None

# --- Config ---
ROOT = Path(__file__).parent.resolve()
VIDEO_DIR = ROOT / "generated_videos"
VIDEO_DIR.mkdir(exist_ok=True)
SAMPLE_ASSET = Path(os.environ.get("SAMPLE_ASSET", str(ROOT / "sample_assets" / "sample.mp4")))

MONGODB_URI = os.environ.get("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not set")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL")

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
REPLICATE_MODEL = os.environ.get("REPLICATE_MODEL", "minimax/video-01")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if openai and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

CLOUDINARY_URL = os.environ.get("CLOUDINARY_URL")
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
WHATSAPP_MAX_BYTES = 16 * 1024 * 1024 - 2000

# --- Logging ---
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(filename=str(LOG_DIR / "server.log"), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# --- Flask ---
app = Flask(__name__)

# --- MongoDB ---
mongo = MongoClient(MONGODB_URI)
db = mongo.get_database(os.environ.get("MONGO_DB_NAME", "peppo_ai"))
sessions_col = db.sessions
cache_col = db.cache
videos_col = db.videos
tasks_col = db.tasks

# Create indexes
try:
    cache_col.create_index([("prompt_norm", ASCENDING)], unique=True)
    sessions_col.create_index([("session_id", ASCENDING)], unique=True)
    tasks_col.create_index([("task_id", ASCENDING)], unique=True)
except Exception:
    logging.exception("Index creation failed")

# --- Twilio ---
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TwilioClient:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- Helper functions ---
from bson import ObjectId

def _serialize_doc(doc):
    if isinstance(doc, list):
        return [_serialize_doc(d) for d in doc]
    if isinstance(doc, dict):
        return {k: _serialize_doc(v) for k, v in doc.items()}
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, datetime):
        return doc.isoformat()
    return doc

def _normalize_prompt(p: str) -> str:
    return " ".join(p.strip().lower().split())

def save_cache(prompt_norm: str, file_path: str, meta: dict = None):
    cache_col.update_one({"prompt_norm": prompt_norm}, {"$set": {"file_path": file_path, "meta": meta or {}, "updated_at": datetime.utcnow()}}, upsert=True)

def load_cache(prompt_norm: str) -> Optional[str]:
    r = cache_col.find_one({"prompt_norm": prompt_norm})
    return r.get("file_path") if r else None

def create_session() -> str:
    sid = uuid.uuid4().hex
    sessions_col.insert_one({"session_id": sid, "created_at": datetime.utcnow(), "messages": []})
    return sid

def append_session(sid: str, role: str, text: str, meta: Optional[dict] = None):
    msg = {"role": role, "text": text, "meta": meta or {}, "ts": datetime.utcnow()}
    sessions_col.update_one({"session_id": sid}, {"$push": {"messages": msg}}, upsert=True)

def get_session(sid: str) -> dict:
    return sessions_col.find_one({"session_id": sid}) or {"session_id": sid, "messages": []}

def record_video(filename: str, path: str, prompt: str, session_id: Optional[str], from_number: Optional[str], cloud_url: Optional[str] = None):
    doc = {"filename": filename, "path": path, "prompt": prompt, "session_id": session_id, "from": from_number, "created_at": datetime.utcnow()}
    if cloud_url:
        doc["cloud_url"] = cloud_url
    videos_col.insert_one(doc)

# --- Cloudinary ---
if cloudinary:
    try:
        if CLOUDINARY_URL:
            cloudinary.config(cloudinary_url=CLOUDINARY_URL)
        else:
            cloudinary.config(
                cloud_name=CLOUDINARY_CLOUD_NAME or None,
                api_key=CLOUDINARY_API_KEY or None,
                api_secret=CLOUDINARY_API_SECRET or None,
                secure=True
            )
    except Exception:
        logging.exception("Cloudinary config failed")

def upload_to_cloudinary(file_path: str, public_id: Optional[str] = None) -> Optional[str]:
    if cloudinary is None:
        return None
    try:
        filesize = Path(file_path).stat().st_size
        if filesize > 10 * 1024 * 1024 and hasattr(cloudinary.uploader, "upload_large"):
            res = cloudinary.uploader.upload_large(file_path, resource_type="video", public_id=public_id, chunk_size=6000000)
        else:
            res = cloudinary.uploader.upload(file_path, resource_type="video", public_id=public_id)
        return res.get("secure_url") or res.get("url")
    except Exception:
        logging.exception("Cloudinary upload failed")
        return None

# --- Replicate helpers ---
FAST_DEFAULTS: Dict[str, Any] = {
    "duration": 5, "fps": 12, "width": 512, "height": 288, "steps": 20, "samples": 1, "guidance_scale": 6
}

def _guess_ext_from_url(url: str) -> str:
    for ext in [".mp4", ".webm", ".gif"]:
        if ext in url: return ext
    return ".mp4"

def _download_to_file(url: str) -> str:
    out = VIDEO_DIR / f"{uuid.uuid4().hex}{_guess_ext_from_url(url)}"
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk: f.write(chunk)
    return str(out)

def _write_bytes_to_file(data: bytes, ext: str = ".mp4") -> str:
    out_path = VIDEO_DIR / f"{uuid.uuid4().hex}{ext}"
    Path(out_path).write_bytes(data)
    return str(out_path)

def _process_replicate_item(item) -> List[str]:
    """
    Try multiple strategies to extract/download a video from a replicate output item.
    Returns list of local file paths (0 or more).
    """
    out_paths: List[str] = []

    # 1) string URL
    try:
        if isinstance(item, str) and item.startswith("http"):
            out_paths.append(_download_to_file(item))
            return out_paths
    except Exception:
        logging.exception("Error processing string item")

    # 2) callable .url()
    try:
        url_callable = getattr(item, "url", None)
        if callable(url_callable):
            try:
                url_val = url_callable()
                if isinstance(url_val, str) and url_val.startswith("http"):
                    out_paths.append(_download_to_file(url_val))
                    return out_paths
                # Sometimes .url() returns a FileOutput-like object with .url property:
                if hasattr(url_val, "url") and isinstance(url_val.url, str) and url_val.url.startswith("http"):
                    out_paths.append(_download_to_file(url_val.url))
                    return out_paths
            except TypeError:
                # some .url attributes are property-like; handled below
                pass
            except Exception:
                logging.exception("Calling item.url() failed")
    except Exception:
        logging.exception("Checking item.url failed")

    # 3) property .url (non-callable)
    try:
        url_prop = getattr(item, "url", None)
        if isinstance(url_prop, str) and url_prop.startswith("http"):
            out_paths.append(_download_to_file(url_prop))
            return out_paths
    except Exception:
        logging.exception("item.url property check failed")

    # 4) .read() -> bytes
    try:
        read_fn = getattr(item, "read", None)
        if callable(read_fn):
            data = read_fn()
            if isinstance(data, (bytes, bytearray)):
                out_paths.append(_write_bytes_to_file(bytes(data), ".mp4"))
                return out_paths
    except Exception:
        logging.exception("Calling item.read() failed")

    # 5) .open() -> file-like
    try:
        open_fn = getattr(item, "open", None)
        if callable(open_fn):
            fobj = open_fn()
            try:
                data = fobj.read()
                if isinstance(data, (bytes, bytearray)):
                    out_paths.append(_write_bytes_to_file(bytes(data), ".mp4"))
                    return out_paths
            finally:
                try:
                    fobj.close()
                except Exception:
                    pass
    except Exception:
        logging.exception("item.open() handling failed")

    # 6) .stream() -> iterable chunks
    try:
        stream_fn = getattr(item, "stream", None)
        if callable(stream_fn):
            stream = stream_fn()
            out_path = VIDEO_DIR / f"{uuid.uuid4().hex}.mp4"
            with open(out_path, "wb") as f:
                for chunk in stream:
                    if isinstance(chunk, (bytes, bytearray)):
                        f.write(chunk)
            out_paths.append(str(out_path))
            return out_paths
    except Exception:
        logging.exception("item.stream() handling failed")

    # 7) download/save methods
    try:
        download_fn = getattr(item, "download", None) or getattr(item, "save", None)
        if callable(download_fn):
            out_path = VIDEO_DIR / f"{uuid.uuid4().hex}.mp4"
            try:
                res = download_fn(str(out_path))
                # if download_fn returns a path or writes file, check
                if isinstance(res, str) and Path(res).exists():
                    out_paths.append(res)
                    return out_paths
                if Path(out_path).exists():
                    out_paths.append(str(out_path))
                    return out_paths
            except Exception:
                logging.exception("item.download/save() failed")
    except Exception:
        logging.exception("item.download/save check failed")

    # 8) dict-like: try common keys
    try:
        if isinstance(item, dict):
            for key in ("url", "output_url", "download_url", "file", "artifact", "data"):
                v = item.get(key)
                if isinstance(v, str) and v.startswith("http"):
                    out_paths.append(_download_to_file(v))
                    return out_paths
                elif isinstance(v, (bytes, bytearray)):
                    out_paths.append(_write_bytes_to_file(bytes(v), ".mp4"))
                    return out_paths
    except Exception:
        logging.exception("dict-like item handling failed")

    # 9) last resort debug logging
    try:
        logging.info("Unrecognized replicate output item type: %s", type(item))
        logging.info("repr(item)[:500]: %s", repr(item)[:500])
        logging.info("dir(item) (partial): %s", ", ".join(dir(item)[:200]))
    except Exception:
        pass

    return out_paths

def call_replicate_minimax(prompt: str, options: Optional[dict] = None) -> List[str]:
    if not REPLICATE_API_TOKEN or replicate is None:
        raise RuntimeError("Replicate not configured")

    norm = _normalize_prompt(prompt)
    cached = load_cache(norm)
    if cached and Path(cached).exists():
        return [cached]

    payload = {**FAST_DEFAULTS, **(options or {}), "prompt": prompt}
    output = replicate.run(REPLICATE_MODEL, input=payload)

    files: List[str] = _process_replicate_item(output)
    if not files:
        raise RuntimeError("No downloadable video returned from replicate")

    save_cache(norm, files[0], {"model": REPLICATE_MODEL})
    return files

# --- Compression ---
def compress_for_whatsapp(in_path: str) -> str:
    size = Path(in_path).stat().st_size
    if size <= WHATSAPP_MAX_BYTES:
        return in_path
    out = str(Path(in_path).with_suffix("")) + f"_c_{uuid.uuid4().hex}.mp4"
    cmd = [FFMPEG_BIN, "-y", "-i", in_path, "-vf", "scale=iw/2:ih/2", "-b:v", "800k", "-preset", "fast", out]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if Path(out).exists() and Path(out).stat().st_size <= WHATSAPP_MAX_BYTES:
            return out
    except Exception:
        logging.exception("Compression failed")
    return in_path

# --- Twilio send ---
def _public_media_url(filename: str) -> Optional[str]:
    return f"{PUBLIC_BASE_URL.rstrip('/')}/media/{filename}" if PUBLIC_BASE_URL else None

def send_twilio_message(to_whatsapp: str, text: str, filename: Optional[str] = None, media_url: Optional[str] = None) -> bool:
    if not twilio_client:
        return False
    try:
        if media_url:
            twilio_client.messages.create(body=text, from_=TWILIO_WHATSAPP_FROM, to=to_whatsapp, media_url=[media_url])
        elif filename:
            url = _public_media_url(filename)
            if url:
                twilio_client.messages.create(body=text, from_=TWILIO_WHATSAPP_FROM, to=to_whatsapp, media_url=[url])
            else:
                twilio_client.messages.create(body=f"{text}\n\nYour file is ready: {filename}", from_=TWILIO_WHATSAPP_FROM, to=to_whatsapp)
        else:
            twilio_client.messages.create(body=text, from_=TWILIO_WHATSAPP_FROM, to=to_whatsapp)
        return True
    except Exception:
        logging.exception("Twilio send failed")
        return False

# --- Bot rules ---
def get_rule_reply(text: str) -> Optional[str]:
    t = text.lower().strip()
    if any(w in t for w in ("hi", "hello", "hey")):
        return "Hi 👋! Send a descriptive prompt like: 'A fox running through neon city night'"
    if t.startswith("/") or "help" in t:
        return "Commands:\n/help - help\n/status - queued tasks\n/history - recent prompts"
    if len(t.split()) < 3:
        return "🤔 Your prompt seems short. Try: 'A sunset over mountains with birds flying'"
    return None

def polish_with_openai(prompt: str, session_msgs: List[Dict[str, Any]]) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY:
        return None
    try:
        msgs = [{"role": "system", "content": "You are a prompt editor: make user's prompt concise for an AI video generator (<=50 words)."}]
        for m in session_msgs[-4:]:
            if m.get("role") == "user":
                msgs.append({"role": "user", "content": m.get("text")})
        msgs.append({"role": "user", "content": prompt})
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=msgs, max_tokens=120, temperature=0.7)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        logging.exception("OpenAI polish failed")
        return None

# --- Worker ---
WORKER: queue.Queue = queue.Queue()
STOP = threading.Event()

def worker_loop():
    logging.info("Worker started")
    while not STOP.is_set():
        try:
            task = WORKER.get(timeout=1)
        except Exception:
            continue
        try:
            tid = task.get("task_id")
            prompt = task.get("prompt")
            sid = task.get("sid")
            from_num = task.get("from")
            options = task.get("options", {}) or {}
            append_session(sid, "assistant", "🎬 Generating your video... this may take a while (usually < 2 minutes)")
            files: List[str] = []
            try:
                files = call_replicate_minimax(prompt, options)
            except Exception as e:
                err = str(e)
                append_session(sid, "assistant", f"❌ Generation failed: {err}")
                tasks_col.update_one({"task_id": tid}, {"$set": {"status": "failed", "error": err, "finished_at": datetime.utcnow()}})
                if from_num: send_twilio_message(from_num, f"Generation failed: {err[:200]}")
                continue

            if not files:
                append_session(sid, "assistant", "❌ Generation returned no files.")
                tasks_col.update_one({"task_id": tid}, {"$set": {"status": "failed", "error": "no files", "finished_at": datetime.utcnow()}})
                if from_num: send_twilio_message(from_num, "No video was generated.")
                continue

            send_path = compress_for_whatsapp(files[0])
            cloud_url = None
            try:
                public_id = f"ai_vids/{uuid.uuid4().hex}"
                cloud_url = upload_to_cloudinary(send_path, public_id)
            except Exception:
                logging.exception("Cloudinary upload error")

            record_video(Path(send_path).name, send_path, prompt, sid, from_num, cloud_url=cloud_url)
            tasks_col.update_one({"task_id": tid}, {"$set": {"status": "done", "finished_at": datetime.utcnow(), "cloud_url": cloud_url}})
            append_session(sid, "assistant", f"✅ Video ready: {Path(send_path).name}")

            if from_num:
                if cloud_url:
                    send_twilio_message(from_num, "✅ Here's your AI-generated video!", media_url=cloud_url)
                elif PUBLIC_BASE_URL:
                    send_twilio_message(from_num, "✅ Here's your AI-generated video!", filename=Path(send_path).name)
                else:
                    send_twilio_message(from_num, "Video generated but not hosted externally. Contact admin.")
        except Exception:
            logging.exception("Task processing failed")
        finally:
            WORKER.task_done()
    logging.info("Worker stopped")

threading.Thread(target=worker_loop, daemon=True).start()

# --- Routes ---
@app.route("/media/<path:fn>", methods=["GET"])
def media(fn):
    p = VIDEO_DIR / fn
    if p.exists():
        return send_file(str(p), mimetype="video/mp4")
    doc = videos_col.find_one({"filename": fn})
    if doc and doc.get("cloud_url"):
        return redirect(doc.get("cloud_url"), code=302)
    return jsonify({"error": "not found"}), 404

@app.route("/api/generate-video", methods=["POST"])
def api_generate():
    body = request.get_json(force=True)
    if not body or "prompt" not in body:
        return jsonify({"error": "provide 'prompt'"}), 400
    prompt = body["prompt"].strip()
    sid = body.get("session_id") or create_session()
    append_session(sid, "user", prompt, {"source": "api"})
    session_msgs = get_session(sid).get("messages", [])
    p2 = polish_with_openai(prompt, session_msgs) or prompt
    tid = uuid.uuid4().hex
    tasks_col.insert_one({"task_id": tid, "prompt": p2, "sid": sid, "from": body.get("from"), "status": "queued", "created_at": datetime.utcnow()})
    WORKER.put({"task_id": tid, "prompt": p2, "sid": sid, "from": body.get("from"), "options": body.get("options", {})})
    return jsonify({"status": "queued", "task_id": tid, "session_id": sid}), 202

@app.route("/api/session-history/<sid>", methods=["GET"])
def api_history(sid):
    s = get_session(sid)
    return jsonify(_serialize_doc(s))

@app.route("/api/list-videos", methods=["GET"])
def api_list_videos():
    docs = list(videos_col.find().sort("created_at", -1).limit(50))
    return jsonify([_serialize_doc(d) for d in docs])

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"ok": True})

@app.route("/whatsapp/webhook", methods=["POST"])
def twilio_webhook():
    data = request.form or request.get_json(silent=True) or {}
    from_number = data.get("From") or data.get("from")
    body = (data.get("Body") or data.get("body") or "").strip()
    if not from_number:
        return jsonify({"error": "missing From"}), 400

    # Normalize command
    cmd = body.lower().strip()

    # --- Handle /status command ---
    if cmd == "/status":
        # Find latest session for this user
        s = sessions_col.find_one({"messages.meta.from": from_number}, sort=[("created_at", -1)])
        if not s:
            reply = "No active sessions found for you."
        else:
            sid = s["session_id"]
            tasks = list(tasks_col.find({"sid": sid}).sort("created_at", -1))
            if not tasks:
                reply = "No tasks found for your session."
            else:
                lines = []
                for t in tasks:
                    status = t.get("status", "queued")
                    prompt = t.get("prompt", "")[:50]
                    created = t.get("created_at").strftime("%Y-%m-%d %H:%M")
                    lines.append(f"[{status}] {prompt} ({created})")
                reply = "\n".join(lines)
        append_session(s.get("session_id") if s else create_session(), "assistant", reply)
        if MessagingResponse:
            resp = MessagingResponse()
            resp.message(reply)
            return str(resp), 200
        send_twilio_message(from_number, reply)
        return jsonify({"reply": reply}), 200

    # --- Handle /history command ---
    if cmd == "/history":
        s = sessions_col.find_one({"messages.meta.from": from_number}, sort=[("created_at", -1)])
        if not s:
            reply = "No session history found for you."
        else:
            sid = s["session_id"]
            videos = list(videos_col.find({"session_id": sid}).sort("created_at", -1))
            if not videos:
                reply = "No videos generated for your session yet."
            else:
                lines = []
                for v in videos:
                    fname = v.get("filename")
                    url = v.get("cloud_url") or _public_media_url(fname)
                    prompt = v.get("prompt", "")[:50]
                    created = v.get("created_at").strftime("%Y-%m-%d %H:%M")
                    lines.append(f"{prompt}\n{url}\n({created})")
                reply = "\n\n".join(lines)
        append_session(s.get("session_id") if s else create_session(), "assistant", reply)
        if MessagingResponse:
            resp = MessagingResponse()
            resp.message(reply)
            return str(resp), 200
        send_twilio_message(from_number, reply)
        return jsonify({"reply": reply}), 200

    # --- Existing prompt handling ---
    r = get_rule_reply(body)
    sid = create_session()
    append_session(sid, "user", body, {"from": from_number})
    if r:
        append_session(sid, "assistant", r)
        if MessagingResponse:
            resp = MessagingResponse()
            resp.message(r)
            return str(resp), 200
        send_twilio_message(from_number, r)
        return jsonify({"reply": r}), 200

    tid = uuid.uuid4().hex
    tasks_col.insert_one({"task_id": tid, "prompt": body, "sid": sid, "from": from_number, "status": "queued", "created_at": datetime.utcnow()})
    WORKER.put({"task_id": tid, "prompt": body, "sid": sid, "from": from_number})
    ack = "🎬 Generating your video... I'll send it here when ready."
    append_session(sid, "assistant", ack)
    if MessagingResponse:
        resp = MessagingResponse(); resp.message(ack); return str(resp), 200
    send_twilio_message(from_number, ack)
    return jsonify({"status": "queued", "reply": ack}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)

