# ScriptSync

Upload a video, pick a language, get a timecoded transcript translated into it — transcription via `faster-whisper`, translation via `deep-translator` (free Google Translate).

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Visit `http://localhost:5000`.

## Push to GitHub
```bash
cd scriptsync
git init
git add .
git commit -m "ScriptSync: video transcription + translation app"
git branch -M main
git remote add origin https://github.com/Ankita2005899/scriptsync.git
git push -u origin main
```
(Create the empty `scriptsync` repo on GitHub first, under Ankita2005899.)

## Deploy on Render
1. New → Web Service → connect the `scriptsync` GitHub repo.
2. Render will detect `render.yaml` automatically (or set manually):
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --timeout 300 --workers 1`
3. Deploy. First build takes a few minutes (moviepy pulls in ffmpeg deps).

## Free tier limits — read before relying on this
Render's free tier is 512MB RAM, no GPU, and spins down when idle (cold start ~30–50s on the next request).
- `WHISPER_MODEL` defaults to `tiny` — fastest, least accurate. `base` is a bit better but slower; only try it if you upgrade RAM.
- Videos are capped at 10 minutes / 60MB in `app.py` (`MAX_CONTENT_LENGTH`, and the duration check in `/api/process`) so a request doesn't get killed mid-run.
- Gunicorn timeout is set to 300s — Render's own proxy may still cut a very long request. For longer/heavier use, move to a paid instance (more RAM = can run `small`/`medium` model, more concurrent requests).

## Project structure
```
scriptsync/
├── app.py              # Flask backend + API
├── templates/
│   └── index.html      # one-page frontend
├── requirements.txt
├── Procfile
├── render.yaml
└── .gitignore
```
