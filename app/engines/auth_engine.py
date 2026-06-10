"""
认证引擎 — 注册 / 登录 / 会话管理
使用 Python 内置 hashlib.pbkdf2_hmac (SHA-256, 260000 iterations)
无需第三方依赖，NIST 推荐标准
"""
import os
import hashlib
import secrets
import logging
import httpx
from datetime import datetime, timezone, timedelta
from typing import Optional

SESSION_TTL_DAYS = 30  # session token 有效期（天）

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


# ── 密码哈希（PBKDF2-HMAC-SHA256）────────────────────────────────────
def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
    return f"pbkdf2:{salt}:{dk.hex()}"


def _verify_password(password: str, hash_str: str) -> bool:
    try:
        _, salt, stored = hash_str.split(":", 2)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
        return secrets.compare_digest(dk.hex(), stored)
    except Exception:
        return False


# ── Supabase helpers ─────────────────────────────────────────────────
def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


def get_user_by_token(token: str) -> Optional[dict]:
    """通过 session_token 查找用户（用于 Bearer 鉴权），同时检查过期时间"""
    if not SUPABASE_URL or not SUPABASE_KEY or not token:
        return None
    try:
        r = httpx.get(
            _url("registrations"),
            headers=_headers(),
            params={"session_token": f"eq.{token}", "select": "*", "limit": "1"},
            timeout=6,
        )
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        user = data[0]
        # ── 检查 session 过期 ─────────────────────────────
        expires_str = user.get("session_expires_at")
        if expires_str:
            try:
                expires_at = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) > expires_at:
                    logger.info(f"session expired for {user.get('phone')}")
                    return None
            except Exception:
                pass  # 解析失败时放行（宽容策略）
        return user
    except Exception as e:
        logger.warning(f"get_user_by_token error: {e}")
        return None


def _get_user_by_phone(phone: str) -> Optional[dict]:
    try:
        r = httpx.get(
            _url("registrations"),
            headers=_headers(),
            params={"phone": f"eq.{phone}", "select": "*", "limit": "1"},
            timeout=8,
        )
        data = r.json()
        return data[0] if isinstance(data, list) and data else None
    except Exception as e:
        logger.warning(f"get_user_by_phone error: {e}")
        return None


# ── 注册 ─────────────────────────────────────────────────────────────
def auth_register(name: str, phone: str, password: str, company: str = "") -> dict:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"ok": False, "error": "数据库未配置"}

    existing = _get_user_by_phone(phone)
    if existing:
        return {"ok": False, "error": "该手机号已注册，请直接登录"}

    pw_hash = _hash_password(password)
    token = secrets.token_hex(24)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)).isoformat()

    try:
        r = httpx.post(
            _url("registrations"),
            headers=_headers(),
            json={
                "name": name,
                "phone": phone,
                "company": company,
                "password_hash": pw_hash,
                "session_token": token,
                "session_expires_at": expires_at,
            },
            timeout=8,
        )
        if r.status_code not in (200, 201):
            logger.warning(f"register insert failed: {r.status_code} {r.text[:200]}")
            return {"ok": False, "error": "注册失败，请稍后重试"}

        return {
            "ok": True,
            "token": token,
            "user": {"name": name, "phone": phone, "company": company},
        }
    except Exception as e:
        logger.warning(f"auth_register error: {e}")
        return {"ok": False, "error": "网络异常，请重试"}


# ── 登录 ─────────────────────────────────────────────────────────────
def auth_login(phone: str, password: str) -> dict:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"ok": False, "error": "数据库未配置"}

    user = _get_user_by_phone(phone)
    if not user:
        return {"ok": False, "error": "手机号未注册，请先注册"}

    pw_hash = user.get("password_hash") or ""
    if not pw_hash:
        return {"ok": False, "error": "该账号未设置密码，请重新注册"}

    if not _verify_password(password, pw_hash):
        return {"ok": False, "error": "密码错误"}

    new_token = secrets.token_hex(24)
    new_expires = (datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)).isoformat()
    try:
        new_count = int(user.get("login_count") or 0) + 1
        httpx.patch(
            _url("registrations"),
            headers={**_headers(), "Prefer": "return=minimal"},
            params={"phone": f"eq.{phone}"},
            json={
                "session_token": new_token,
                "session_expires_at": new_expires,
                "last_login": datetime.now(timezone.utc).isoformat(),
                "login_count": new_count,
            },
            timeout=8,
        )
    except Exception as e:
        logger.warning(f"token refresh error: {e}")

    return {
        "ok": True,
        "token": new_token,
        "user": {
            "name": user.get("name", ""),
            "phone": phone,
            "company": user.get("company", ""),
            "role": user.get("role", "user"),
        },
    }


# ── 退出登录 ──────────────────────────────────────────────────────────
def clear_session_token(phone: str) -> dict:
    """退出登录：清除服务器端 session_token"""
    if not SUPABASE_URL or not SUPABASE_KEY or not phone:
        return {"ok": False, "error": "参数缺失"}
    try:
        httpx.patch(
            _url("registrations"),
            headers={**_headers(), "Prefer": "return=minimal"},
            params={"phone": f"eq.{phone}"},
            json={"session_token": ""},
            timeout=6,
        )
        return {"ok": True}
    except Exception as e:
        logger.warning(f"clear_session_token error: {e}")
        return {"ok": False, "error": str(e)}


# ── 设置用户角色（管理员专用）────────────────────────────────────────
def set_user_role(phone: str, role: str) -> dict:
    """将指定手机号用户的 role 字段设为 admin 或 user"""
    if not SUPABASE_URL or not SUPABASE_KEY or not phone:
        return {"ok": False, "error": "参数缺失"}
    try:
        r = httpx.patch(
            _url("registrations"),
            headers={**_headers(), "Prefer": "return=minimal"},
            params={"phone": f"eq.{phone}"},
            json={"role": role},
            timeout=6,
        )
        if r.status_code in (200, 201, 204):
            return {"ok": True, "phone": phone, "role": role}
        return {"ok": False, "error": f"Supabase 返回 {r.status_code}: {r.text[:100]}"}
    except Exception as e:
        logger.warning(f"set_user_role error: {e}")
        return {"ok": False, "error": str(e)}
