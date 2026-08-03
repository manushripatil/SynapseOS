# Deploy SynapseOS

You can share the **live visual simulator** on the internet.

---

## Option A — Local (already works)

```bash
cd synapseos
pip3 install -r requirements.txt
python3 visual/app.py
```

Open: http://127.0.0.1:5050

---

## Option B — Deploy free on Render.com (recommended)

1. Create a free account at https://render.com  
2. Push this `synapseos` folder to a **GitHub** repo  
3. In Render: **New → Web Service** → connect that repo  
4. Settings:
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt gunicorn`
   - **Start command:** `gunicorn -b 0.0.0.0:$PORT --workers 1 --threads 4 visual.wsgi:app`
5. Click **Deploy**

You get a public URL like: `https://synapseos-xxxx.onrender.com`

---

## Option C — Railway.app

1. https://railway.app → New Project → Deploy from GitHub  
2. Root directory: `synapseos` (if repo is parent)  
3. Railway detects `Procfile` automatically  
4. Add env if needed: `PORT` is set by Railway  

---

## Option D — Docker (any VPS)

```bash
cd synapseos
docker build -t synapseos .
docker run -p 5050:5050 synapseos
```

---

## What gets deployed

The **visual simulator** (Phase 1–4 engine in the browser):

- Live cell fabric  
- Start / damage / killer demo  
- Role colours, maturity, energy  

CLI experiments (`demo_recovery.py`, `runner.py`) stay on your machine — they are research tools, not the public demo.

---

## Before you push to GitHub

```bash
cd synapseos
git init
git add .
git commit -m "SynapseOS: self-organising neuromorphic fabric"
# create empty repo on GitHub, then:
git remote add origin https://github.com/YOUR_USER/SynapseOS.git
git push -u origin main
```

Do **not** commit huge `output/` plot folders if you add them later.
