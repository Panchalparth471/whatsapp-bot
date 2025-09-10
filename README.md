# WhatsApp AI Video Bot — README

**Live deployment:** `https://whatsapp-bot-kg4m.onrender.com`

**Purpose:** Convert an AI video generator into an interactive WhatsApp bot. The bot receives prompts via WhatsApp (Twilio), optionally polishes them (OpenAI), queues jobs, calls an AI model (Replicate), compresses results (FFmpeg), uploads to Cloudinary (or serves locally), and returns videos to users via Twilio.

---

## Highlights / Features

* **WhatsApp (Twilio) Integration** — inbound webhook + outgoing messages using `twilio.rest`.
* **Queue & Worker** — in-process `queue.Queue` with background worker(s); `WORKER_COUNT` configurable.
* **AI Generation (Replicate)** — calls `REPLICATE_MODEL` to generate videos from prompts.
* **Prompt Polish (OpenAI)** — optional prompt refinement before generation.
* **Caching** — normalized-prompt caching in MongoDB to avoid duplicate work.
* **Media Hosting** — Cloudinary upload (preferred) + fallback to app’s `/media/<filename>` using `PUBLIC_BASE_URL`.
* **Compression for WhatsApp** — FFmpeg-based compression to meet WhatsApp/Twilio media size limits.
* **Persistence (MongoDB)** — sessions, tasks, videos, cache stored in collections.
* **Commands** — `/status`, `/history`, `/help`.
* **Robust error handling & fallbacks** — Cloudinary fallback, text fallback when media delivery fails.

---

## Architecture

```

```

ASCII alternative:
`WhatsApp User -> Twilio -> Flask App -> Queue -> Worker -> Replicate -> Cloudinary -> Twilio -> User`
Fallback: `Worker -> Local /media/ -> Flask /media/<filename> -> Twilio -> User`

---

## Quickstart (for Render deployment)

deployed app is:

```
https://whatsapp-bot-kg4m.onrender.com
```

### Required environment variables (set in Render → Environment)

> **Important:** Do **not** commit secrets to GitHub. Use Render's environment/secret UI.

```
# Database
MONGODB_URI="mongodb+srv://<user>:<pass>@cluster0.mongodb.net/Peppo"
MONGO_DB_NAME=peppo_ai

# Twilio
TWILIO_ACCOUNT_SID=ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_FROM=whatsapp:+1415XXXXXXX

# Deployment
PUBLIC_BASE_URL=https://whatsapp-bot-kg4m.onrender.com

# Replicate
REPLICATE_API_TOKEN=your_replicate_api_token
REPLICATE_MODEL=minimax/video-01

# Cloudinary (optional)
CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME

# OpenAI (optional)
OPENAI_API_KEY=sk-...

# FFmpeg
FFMPEG_BIN=ffmpeg

# Runtime options
WORKER_COUNT=2
WHATSAPP_MAX_BYTES=16700000
```

---

## Run & test (endpoints)

### Health check

```
GET https://whatsapp-bot-kg4m.onrender.com/health
```

### Test-send (server → WhatsApp via Twilio)

Use `/test-send` to confirm outbound sending:

```bash
curl -X POST "https://whatsapp-bot-kg4m.onrender.com/test-send" \
  -d "to=whatsapp:+91XXXXXXXXXX" \
  -d "message=Hello from deployed bot!"
```

### Queue a generation via API

```bash
curl -X POST "https://whatsapp-bot-kg4m.onrender.com/api/generate-video" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A fox running through neon city night","from":"whatsapp:+91XXXXXXXXXX"}'
```

### Endpoints summary

* `GET /` — health summary
* `GET /health` — detailed health
* `POST /api/generate-video` — create a task (JSON `{ "prompt": "...", "from": "whatsapp:+<number>" }`)
* `GET /api/session-history/<sid>` — session messages
* `GET /api/list-videos` — recent videos list
* `POST /whatsapp/webhook` — Twilio webhook (handles inbound WhatsApp messages & media)
* `GET /media/<filename>` — serve generated videos (fallback)

---

## How to test the Twilio webhook with **Postman** (and Twilio Sandbox)

You can test in two ways:

### Option A — Real flow via Twilio Sandbox (recommended)

1. Go to Twilio Console → Messaging → **Try it out → WhatsApp Sandbox**.
2. Set **When a message comes in** webhook to:

```
https://whatsapp-bot-kg4m.onrender.com/whatsapp/webhook
```

(Method: POST)
3\. In WhatsApp, send `join <sandbox-code>` to the Twilio sandbox number to link your phone.
4\. Message the sandbox number — Twilio will forward messages to your Render webhook and you’ll get responses.

### Option B — Simulate Twilio POST via Postman (quick validation)

> If `X-Twilio-Signature` validation is enabled in your app, either temporarily disable validation for testing or compute a valid signature (below).

1. Create a `POST` request:

```
POST https://whatsapp-bot-kg4m.onrender.com/whatsapp/webhook
```

2. Body → `form-data`:

* `From` = `whatsapp:+9199XXXXXXXX`
* `Body` = `Create a video of a cat playing piano in space`
* Optional: `MediaUrl0` = `https://example.com/sample.mp4`
* Optional: `MediaContentType0` = `video/mp4`

3. (Optional) Header:

* `X-Twilio-Signature: <computed-signature>` — only required if signature validation enforcement is active.

4. Send request. Inspect the response and Render logs for parsing and task enqueue confirmation.

### Compute a Twilio signature for Postman (Python example)

```python
from twilio.request_validator import RequestValidator

auth_token = "YOUR_TWILIO_AUTH_TOKEN"
validator = RequestValidator(auth_token)

url = "https://whatsapp-bot-kg4m.onrender.com/whatsapp/webhook"
params = {
    "From": "whatsapp:+9199XXXXXXXX",
    "Body": "Create a video of a cat playing piano in space"
}
sig = validator.compute_signature(url, params)
print(sig)  # set this value as X-Twilio-Signature in Postman
```

> Keep `TWILIO_AUTH_TOKEN` secret and **do not** commit it.

---

## Supported WhatsApp commands

* `/help` — list commands & examples
* `/status` — latest tasks & statuses for your session
* `/history` — recently generated videos & links

Typical conversation:

* User: `A fox running through neon city night`
* Bot: `🎬 Generating your video... I'll send it here when ready.`
* Later: `✅ Here's your AI-generated video!` + media or link

---

## Data model (MongoDB collections)

* **sessions** — `{ session_id, created_at, messages: [{role, text, meta, ts}] }`
* **tasks** — `{ task_id, prompt, sid, from, status, created_at, finished_at, file_path, cloud_url, error }`
* **videos** — `{ filename, path, prompt, session_id, from, created_at, cloud_url }`
* **cache** — `{ prompt_norm, file_path, meta, updated_at }`

---

## Security & operational notes

* **Never commit secrets** (`.env`) to source control. Use Render environment variables.
* **Enable Twilio request validation** in production — reject requests with invalid `X-Twilio-Signature`.
* **Rate-limit** by phone number (e.g., `10/min`) to avoid abuse — use `flask-limiter`.
* **Ensure FFmpeg is present** in the runtime (Render may need Docker image with ffmpeg installed).
* **PUBLIC\_BASE\_URL** must be `https://whatsapp-bot-kg4m.onrender.com` in your Render env so Twilio can fetch fallback `/media/<filename>` URLs.
* **Add monitoring** (Sentry / Prometheus / Render logs) for production observability.

---

## Troubleshooting

**Twilio can't fetch media**

* Verify `PUBLIC_BASE_URL` is HTTPS and accessible. Prefer Cloudinary `secure_url` when possible.

**FFmpeg not found / compression failing**

* Use a Docker image that installs `ffmpeg` (example Dockerfile below).

**Replicate API fails or times out**

* Confirm `REPLICATE_API_TOKEN` & `REPLICATE_MODEL`. Implement retry/backoff if needed.

**Webhook signature invalid**

* Ensure the URL used to compute/validate the signature equals Twilio’s public URL. When behind proxies, consider reconstructing using `X-Forwarded-Proto` and `X-Forwarded-Host` or use a fixed public URL for validation.

---

## Dockerfile example (includes FFmpeg)

Use this if you want to deploy on Render with ffmpeg available:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg build-essential --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1
CMD ["python", "server.py"]
```

---

## Deliverables checklist for Peppo Round-2

* ✅ Working WhatsApp bot reachable at `https://whatsapp-bot-kg4m.onrender.com`
* ✅ Twilio sandbox or WhatsApp-enabled Twilio number configured
* ✅ `/status`, `/history`, `/help` commands implemented
* ✅ Queueing + background processing (configurable worker count)
* ✅ Compression before sending to WhatsApp (FFmpeg)
* ✅ Cloudinary upload with fallback to `/media/<filename>` (PUBLIC\_BASE\_URL)
* ✅ README (this file) + architecture diagram
* ✅ Demo video (2–3 mins) showing `/status`, prompt → ack → video delivered
* ✅ Clean branch for Round 2 (commits showing integration work)

---

## Example `.env` (DO NOT commit; replace placeholders with your values)

```text
# Database
MONGODB_URI="mongodb+srv://<user>:<password>@cluster0.mongodb.net/Peppo"
MONGO_DB_NAME=peppo_ai

# Twilio
TWILIO_ACCOUNT_SID=ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886

# Deployment
PUBLIC_BASE_URL=https://whatsapp-bot-kg4m.onrender.com

# Replicate
REPLICATE_API_TOKEN=your_replicate_api_token
REPLICATE_MODEL=minimax/video-01

# Cloudinary (optional)
CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME

# OpenAI (optional)
OPENAI_API_KEY=sk-...

# FFmpeg
FFMPEG_BIN=ffmpeg

# Runtime
WORKER_COUNT=2
WHATSAPP_MAX_BYTES=16700000
```


---

## Tips & recommended improvements

* Move the in-process queue to a persistent job system (Celery / Redis Queue / RQ) for production reliability.
* Add a small worker pool (set `WORKER_COUNT`) or use separate worker service(s).
* Record task `processing` state and attempt counts to support retries and crash recovery.
* Add more granular progress messages (e.g., "Worker started processing your task").
* Add request-rate limiting and prompt-length caps to prevent abuse.
* Log Twilio delivery SIDs and optionally handle delivery receipt webhooks.

---

## Demo recording checklist (2–3 minutes)

1. Show the Render dashboard with service running and `/health` returning `healthy`.
2. Show Twilio Sandbox settings with inbound webhook set to `https://whatsapp-bot-kg4m.onrender.com/whatsapp/webhook`.
3. Demonstrate joining the Twilio sandbox from a WhatsApp account (`join <code>`).
4. Send a sample prompt from WhatsApp: `A cat playing piano in space`.
5. Show the bot responding with acknowledgement.
6. Show the final video arriving in WhatsApp (or show the message with the Cloudinary link).
7. Demonstrate `/status` and `/history`.
8. Optionally show server logs of task processing and Cloudinary upload.

---

## Submission & how to share the bot with reviewers

**If using Twilio Sandbox**

* Share Twilio Sandbox phone number and join code (from Twilio console). Example message:

> Demo: WhatsApp AI Video Bot
>
> * Sandbox number: `+1XXXXXXXXXX`
> * Join code: `join <your-sandbox-code>`
> * Hosted at: `https://whatsapp-bot-kg4m.onrender.com`
>   Steps: Send `join <code>` to the sandbox number in WhatsApp; then message a prompt.

**If using a production Twilio WhatsApp number**

* Share the WhatsApp number (e.g., `+1XXXXXXXX`) and instruct reviewers to message it directly.

---

## License

MIT

---

* a `docker-compose.yml` and `render.yaml` (Render + Docker) ready for copy/paste,
* a Postman collection JSON (with sample requests) you can import, or
* the code patches to **(A)** enable strict Twilio signature validation, **(B)** inbound media download handling, and **(C)** worker-count thre
