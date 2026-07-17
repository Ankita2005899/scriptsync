import os
import shutil
import tempfile
import threading
import uuid
import time

import requests
from flask import Flask, render_template, request, jsonify
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import moviepy.editor as mp

app = Flask(__name__)

# Note: this does NOT change Render free tier's 512MB RAM ceiling.
# A very large video can still crash/hang the worker during transcription.
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

# If set, /api/process forwards the video to a GPU-backed Colab server
# instead of processing it locally on Render's slow free-tier CPU.
# Set this in Render -> Environment -> COLAB_BACKEND_URL to the ngrok URL
# printed by the Colab notebook's last cell (no trailing slash).
# If unset, empty, or unreachable, this app falls back to local CPU
# processing automatically -- nothing breaks if you don't have Colab running.
COLAB_BACKEND_URL = os.environ.get("COLAB_BACKEND_URL", "").rstrip("/")

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")  # tiny/base = free-tier safe
_model = None
_model_lock = threading.Lock()


def get_model():
    """Load the whisper model once, lazily (keeps cold-start fast)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def _preload_model():
    """
    Warm up the model as soon as the worker boots, in a background thread,
    instead of on the first user request. Render's free tier wipes any
    cached model files on every restart/redeploy, so this download/load
    cost happens once per deploy no matter what — this just moves it out
    of a user's first request and into server startup instead.
    """
    try:
        get_model()
    except Exception:
        # If preloading fails for any reason, get_model() will just retry
        # lazily on the first real request instead.
        pass


threading.Thread(target=_preload_model, daemon=True).start()


LANGUAGES = {
    "Hindi": "hi", "English": "en", "Marathi": "mr", "Bengali": "bn",
    "Tamil": "ta", "Telugu": "te", "Gujarati": "gu", "Kannada": "kn",
    "Malayalam": "ml", "Punjabi": "pa", "Urdu": "ur",
    "Spanish": "es", "French": "fr", "German": "de", "Portuguese": "pt",
    "Russian": "ru", "Chinese (Simplified)": "zh-CN", "Japanese": "ja",
    "Korean": "ko", "Arabic": "ar",
}

# In-memory job store. Fine for a single free-tier worker;
# jobs are lost on redeploy/restart, which is acceptable here.
JOBS = {}
JOBS_LOCK = threading.Lock()

# Drop jobs older than this so memory doesn't grow forever (seconds)
JOB_TTL_SECONDS = 60 * 60  # 1 hour

# Ordered stages used to compute a progress percentage on the frontend
STAGE_ORDER = ["queued", "extracting_audio", "loading_model", "transcribing", "translating", "done"]


def _cleanup_old_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with JOBS_LOCK:
        stale = [jid for jid, job in JOBS.items() if job.get("created_at", 0) < cutoff]
        for jid in stale:
            JOBS.pop(jid, None)


def _set_job(job_id, **fields):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def _try_colab_backend(job_id, video_path, target_lang):
    """
    Attempts to forward the video to the Colab GPU backend. Returns True if
    it succeeded and the job was completed this way, False if it should
    fall back to local CPU processing instead (Colab not configured, not
    reachable, or it returned an error).
    """
    if not COLAB_BACKEND_URL:
        return False

    _set_job(job_id, status="transcribing")  # Colab is fast enough that finer stages aren't very useful
    try:
        with open(video_path, "rb") as f:
            resp = requests.post(
                f"{COLAB_BACKEND_URL}/api/process",
                files={"video": (os.path.basename(video_path), f)},
                data={"language": target_lang},
                timeout=180,  # Colab GPU should be much faster than this
            )
        if resp.status_code != 200:
            # Colab reachable but returned an error -- fall back to local.
            return False

        data = resp.json()
        if "segments" not in data:
            return False

        _set_job(job_id, status="translating")  # brief, mostly for UI continuity
        _set_job(
            job_id,
            status="done",
            result={
                "detected_language": data.get("detected_language", "unknown"),
                "target_language": data.get("target_language", target_lang),
                "segments": data["segments"],
            },
        )
        return True

    except (requests.exceptions.RequestException, ValueError):
        # Colab notebook not running, tunnel expired, network hiccup, or
        # bad JSON -- silently fall back to local processing below.
        return False


def run_job(job_id, video_path, target_lang, tmp_dir):
    """Runs in a background thread. Does the actual heavy lifting."""
    try:
        if _try_colab_backend(job_id, video_path, target_lang):
            return  # Colab handled it successfully; nothing more to do.

        # Fallback: process locally on Render's CPU, same as before.
        _set_job(job_id, status="extracting_audio")

        audio_path = os.path.join(tmp_dir, "audio.wav")
        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()

        if _model is None:
            _set_job(job_id, status="loading_model")

        model = get_model()

        _set_job(job_id, status="transcribing")
        segments_gen, info = model.transcribe(
            audio_path,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        raw_segments = [
            {"start": round(seg.start, 2), "end": round(seg.end, 2), "original": seg.text.strip()}
            for seg in segments_gen if seg.text.strip()
        ]

        _set_job(job_id, status="translating")

        # Batch translation instead of one API call per segment.
        # deep_translator's GoogleTranslator has a ~5000 char limit per call,
        # so we join segments with a unique delimiter and split the result
        # back apart, cutting dozens of network round-trips down to just a few.
        translator = GoogleTranslator(source="auto", target=target_lang)
        DELIM = "\n|||\n"
        BATCH_CHAR_LIMIT = 4000

        batches = []
        current_batch, current_len = [], 0
        for seg in raw_segments:
            seg_len = len(seg["original"]) + len(DELIM)
            if current_batch and current_len + seg_len > BATCH_CHAR_LIMIT:
                batches.append(current_batch)
                current_batch, current_len = [], 0
            current_batch.append(seg)
            current_len += seg_len
        if current_batch:
            batches.append(current_batch)

        segments = []
        for batch in batches:
            joined = DELIM.join(s["original"] for s in batch)
            try:
                translated_joined = translator.translate(joined)
                parts = translated_joined.split(DELIM.strip())
                if len(parts) != len(batch):
                    # Delimiter got mangled by translation; fall back to
                    # per-segment translation for just this batch.
                    parts = []
                    for s in batch:
                        try:
                            parts.append(translator.translate(s["original"]))
                        except Exception:
                            parts.append(s["original"])
            except Exception:
                parts = [s["original"] for s in batch]

            for seg, translated in zip(batch, parts):
                segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "original": seg["original"],
                    "translated": translated.strip(),
                })

        _set_job(
            job_id,
            status="done",
            result={
                "detected_language": info.language,
                "target_language": target_lang,
                "segments": segments,
            },
        )

    except Exception as e:
        _set_job(job_id, status="error", error=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGES)


@app.route("/api/ready")
def api_ready():
    return jsonify({"model_ready": _model is not None})


@app.route("/api/languages")
def api_languages():
    return jsonify(LANGUAGES)


@app.route("/api/process", methods=["POST"])
def api_process():
    _cleanup_old_jobs()

    if "video" not in request.files:
        return jsonify({"error": "No video file received."}), 400

    video = request.files["video"]
    target_lang = request.form.get("language", "hi")

    if video.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, video.filename)
    video.save(video_path)

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "created_at": time.time()}

    thread = threading.Thread(
        target=run_job,
        args=(job_id, video_path, target_lang, tmp_dir),
        daemon=True,
    )
    thread.start()

    # Return immediately so Render's proxy never has to hold this request open.
    return jsonify({"job_id": job_id}), 202


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None:
        return jsonify({"error": "Unknown or expired job."}), 404

    status = job.get("status")
    try:
        stage_index = STAGE_ORDER.index(status)
    except ValueError:
        stage_index = 0
    progress_pct = round((stage_index / (len(STAGE_ORDER) - 1)) * 100)

    response = {"status": status, "progress": progress_pct}
    if status == "done":
        response["result"] = job.get("result")
    elif status == "error":
        response["error"] = job.get("error")

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))