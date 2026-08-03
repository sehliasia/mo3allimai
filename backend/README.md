# Mo3allimAI API (Windows CMD)

```cmd
cd backend
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
```

Swagger: http://127.0.0.1:8000/docs. Run tests with `pytest`.

Create the first admin: `python -m app.scripts.create_admin`.

Logout is currently client-side: remove the bearer token. Refresh tokens, rotation, revocation and HttpOnly cookies can be added later.
