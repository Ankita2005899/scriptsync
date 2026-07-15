import os
import shutil
import tempfile

from flask import Flask, render_template, request, jsonify
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import moviepy.editor as mp

app = Flask(__name__)

# Raised from 60MB — no longer capping uploads tightly.
# Note: this does NOT change Render free tier's 512MB RAM ceiling.
# A very large video can still crash/hang the worker during transcription.
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")  # tiny/base = free-tier safe
_model = None


def get_model():
    """Load the whisper model once, lazily (keeps cold-start fast)."""
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


LANGUAGES = {
    "Hindi": "hi", "English": "en", "Marathi": "mr", "Bengali": "bn",
    "Tamil": "ta", "Telugu": "te", "Gujarati": "gu", "Kannada": "kn",
    "Malayalam": "ml", "Punjabi": "pa", "Urdu": "ur",
    "Spanish": "es", "French": "fr", "German": "de", "Portuguese": "pt",
    "Russian": "ru", "Chinese (Simplified)": "zh-CN", "Japanese": "ja",
    "Korean": "ko", "Arabic": "ar",
}


@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGES)


@app.route("/api/languages")
def api_languages():
    return jsonify(LANGUAGES)


@app.route("/api/process", methods=["POST"])
def api_process():
    if "video" not in request.files:
        return jsonify({"error": "No video file received."}), 400

    video = request.files["video"]
    target_lang = request.form.get("language", "hi")

    if video.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    tmp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(tmp_dir, video.filename)
        video.save(video_path)

        audio_path = os.path.join(tmp_dir, "audio.wav")
        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()

        model = get_model()
        segments_gen, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

        translator = GoogleTranslator(source="auto", target=target_lang)
        segments = []
        for seg in segments_gen:
            original = seg.text.strip()
            if not original:
                continue
            try:
                translated = translator.translate(original)
            except Exception:
                translated = original
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "original": original,
                "translated": translated,
            })

        return jsonify({
            "detected_language": info.language,
            "target_language": target_lang,
            "segments": segments,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))