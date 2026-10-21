"""
Login System — FastAPI + Jinja2 (no database)  ·  v2
====================================================

A clean, production-shaped authentication demo:

    * User registration with full server-side validation
    * Password hashing with bcrypt (direct, salted — no passlib needed)
    * Stateless JWT authentication (PyJWT, HS256, 30-minute expiry)
    * Token transported in an HttpOnly cookie (JavaScript cannot read it)
    * Brute-force protection: temporary lockout after repeated failures
    * Protected dashboard route + logout + /health endpoint

Auth flow
---------
    POST /register   ->  validate input, store user with HASHED password
    POST /login      ->  check lockout, verify bcrypt hash, sign JWT, set cookie
    GET  /dashboard  ->  decode & verify cookie JWT, load user
    POST /logout     ->  delete the cookie
    GET  /health     ->  uptime check (used by Render)

The "database" is an in-memory dict (the assignment excludes databases),
so every account is wiped when the server restarts — by design.

Run locally
-----------
    pip install -r requirements.txt
    uvicorn main:app --reload
    # open http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt
import jwt
from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "dev-only-secret-change-me-3f8a1c92b7d64e05a1c9",  # override in production!
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
COOKIE_NAME = "access_token"
# Set COOKIE_SECURE=true when serving over HTTPS (e.g. Render / Railway).
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 2. App, static files and Jinja templates
# ---------------------------------------------------------------------------
app = FastAPI(title="SecureAuth — Login System")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ---------------------------------------------------------------------------
# 3. Password hashing — bcrypt directly (salt generated automatically)
#    bcrypt only uses the first 72 BYTES of input, so registration caps
#    the length (see validate_password_strength) to avoid errors.
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Turn a plaintext password into a salted bcrypt hash (stored as text)."""
    hashed: bytes = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time check of a plaintext password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # Malformed hash in storage — treat as failed verification, never crash.
        return False


# ---------------------------------------------------------------------------
# 4. In-memory user store ("database is not included")
# ---------------------------------------------------------------------------
# users_db["john"] = {"email": "...", "password": "<bcrypt hash>", "created_at": ...}
users_db: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------
USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,19}$")
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


def validate_password_strength(password: str) -> list[str]:
    """Return a list of every rule the password fails (empty list == strong)."""
    problems: list[str] = []
    if len(password) < 8:
        problems.append("Password must be at least 8 characters long.")
    if len(password.encode("utf-8")) > 72:
        problems.append("Password must be at most 72 characters long.")
    if not re.search(r"[A-Z]", password):
        problems.append("Password must contain at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        problems.append("Password must contain at least one lowercase letter.")
    if not re.search(r"\d", password):
        problems.append("Password must contain at least one number.")
    if not re.search(r"[^A-Za-z0-9]", password):
        problems.append("Password must contain at least one special character.")
    return problems


def validate_registration(
    username: str, email: str, password: str, confirm_password: str
) -> list[str]:
    """Validate the whole registration form; returns human-readable errors."""
    errors: list[str] = []

    if not USERNAME_RE.match(username):
        errors.append(
            "Username must be 3-20 characters, start with a letter and use "
            "only letters, numbers or underscores."
        )
    if username.lower() in users_db:
        errors.append("That username is already taken.")
    if not EMAIL_RE.match(email):
        errors.append("Please enter a valid email address.")

    errors.extend(validate_password_strength(password))

    if password != confirm_password:
        errors.append("Passwords do not match.")

    return errors


# ---------------------------------------------------------------------------
# 6. Brute-force protection — lock the account for a while after
#    MAX_FAILED_LOGINS consecutive failures (in-memory, per process).
# ---------------------------------------------------------------------------
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 5 * 60  # 5 minutes

# username -> (failed_count, locked_until_timestamp)
failed_logins: dict[str, tuple[int, float]] = {}


def lockout_minutes_left(username: str) -> int:
    """Return how many minutes the account is still locked (0 = not locked)."""
    record = failed_logins.get(username)
    if record is None:
        return 0
    _count, locked_until = record
    if locked_until and time.time() < locked_until:
        return int((locked_until - time.time()) // 60) + 1
    return 0


def record_failed_login(username: str) -> None:
    """Count a failed attempt; trigger a lockout at the threshold."""
    count, locked_until = failed_logins.get(username, (0, 0.0))
    count += 1
    if count >= MAX_FAILED_LOGINS:
        locked_until = time.time() + LOCKOUT_SECONDS
        count = 0  # counter resets; the timestamp keeps the lock active
    failed_logins[username] = (count, locked_until)


def clear_failed_logins(username: str) -> None:
    """Reset the counter after a successful login."""
    failed_logins.pop(username, None)


# ---------------------------------------------------------------------------
# 7. JWT helpers
# ---------------------------------------------------------------------------
def create_access_token(username: str) -> str:
    """Sign a JWT containing the username ('sub') plus iat / exp claims."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Verify signature + expiry. Returns the payload, or None if invalid."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None  # token expired
    except jwt.InvalidTokenError:
        return None  # tampered / malformed token


def get_current_user(request: Request) -> Optional[dict]:
    """Resolve the logged-in user from the JWT cookie (None if anonymous)."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None

    payload = decode_token(token)
    if payload is None:
        return None

    username = payload.get("sub")
    user = users_db.get(username)  # None if the store was reset (restart)
    if user is None:
        return None

    return {"username": username, "user": user, "payload": payload}


# ---------------------------------------------------------------------------
# 8. Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    """Send visitors to the dashboard if logged in, otherwise to /login."""
    if get_current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/login", status_code=303)


# -- Health check ------------------------------------------------------------
@app.get("/health")
async def health():
    """Tiny JSON endpoint for uptime monitors (Render pings this)."""
    return {"status": "ok"}


# -- Registration -----------------------------------------------------------
@app.get("/register")
async def register_page(request: Request):
    if get_current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        request, "register.html", {"errors": [], "form_username": "", "form_email": ""}
    )


@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    username = username.strip()
    email = email.strip().lower()

    errors = validate_registration(username, email, password, confirm_password)
    if errors:
        # Re-render the form so the user keeps what they already typed.
        return templates.TemplateResponse(
            request,
            "register.html",
            {"errors": errors, "form_username": username, "form_email": email},
            status_code=400,
        )

    users_db[username.lower()] = {
        "email": email,
        "password": hash_password(password),  # plaintext is NEVER stored
        "created_at": datetime.now(timezone.utc),
    }
    return RedirectResponse("/login?registered=true", status_code=303)


# -- Login ------------------------------------------------------------------
@app.get("/login")
async def login_page(
    request: Request,
    error: Optional[str] = None,
    msg: Optional[str] = None,
    registered: bool = False,
):
    if get_current_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        request,
        "login.html",
        {"error": error, "msg": msg, "registered": registered, "form_username": ""},
    )


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    username = username.strip().lower()

    # 1. Brute-force guard: block this username while it is locked out.
    minutes = lockout_minutes_left(username)
    if minutes:
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "error": (
                    f"Too many failed attempts. This account is locked for "
                    f"about {minutes} more minute(s)."
                ),
                "msg": None,
                "registered": False,
                "form_username": username,
            },
            status_code=429,
        )

    # 2. Verify credentials against the stored bcrypt hash.
    user = users_db.get(username)
    if user is None or not verify_password(password, user["password"]):
        record_failed_login(username)
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "error": "Invalid username or password.",
                "msg": None,
                "registered": False,
                "form_username": username,
            },
            status_code=401,
        )

    # 3. Success: reset the failure counter, sign the JWT, set the cookie.
    clear_failed_logins(username)
    token = create_access_token(username)
    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,      # JavaScript cannot read it (XSS protection)
        samesite="lax",     # basic CSRF protection
        secure=COOKIE_SECURE,
        path="/",
    )
    return response


# -- Dashboard (protected route) ---------------------------------------------
def _fmt(dt: datetime) -> str:
    return dt.strftime("%b %d, %Y · %H:%M UTC")


@app.get("/dashboard")
async def dashboard(request: Request):
    current = get_current_user(request)
    if current is None:
        return RedirectResponse(
            "/login?error=Please+log+in+to+view+your+dashboard", status_code=303
        )

    payload = current["payload"]
    user = current["user"]

    now = datetime.now(timezone.utc)
    issued_at = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
    expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    seconds_left = max(0, int((expires_at - now).total_seconds()))
    total_seconds = max(1, int((expires_at - issued_at).total_seconds()))

    # session_percent is passed as plain data; the template applies it via
    # a data-attribute + two lines of JS (keeps inline CSS 100% static,
    # so editors do not flag the Jinja placeholder as invalid CSS).
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "username": current["username"],
            "email": user["email"],
            "member_since": _fmt(user["created_at"]),
            "issued_at": _fmt(issued_at),
            "expires_at": _fmt(expires_at),
            "minutes_left": seconds_left // 60,
            "session_percent": round(seconds_left / total_seconds * 100),
        },
    )


# -- Logout -------------------------------------------------------------------
@app.post("/logout")
async def logout():
    response = RedirectResponse("/login?msg=You+have+been+logged+out", status_code=303)
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )