# Kartly Hybrid Order Support Assistant

Work in progress.

## Run so far

Copy the environment file, install dependencies, seed the database, and start the API:

```bash
cp .env.example .env
pip install -r requirements.txt
python -m scripts.seed
uvicorn app.main:app --reload
```

Open `http://localhost:8000/health` to check the service.

Alternatively, run `docker compose up --build` after creating `.env`.
