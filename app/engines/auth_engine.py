"""
认证引擎 — 注册 / 登录 / 会话管理
使用 bcrypt 密码哈希，会话 token 存储在 Supabase registrations 表
"""
import os
import secrets
import logging
import httpx
from passlib.context import CryptContext

logger = logging.getLogger(__name__)
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def _headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def _url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


def _get_user_by_phone(phone: str) -> dict | None:
    """通过手机号查找用户，返回第一条记录或 None"""
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


def auth_register(name: str, phone: str, password: str, company: str = "") -> dict:
    """
    注册新账号。
    - 手机号已存在 → 报错
    - 成功 → 返回 {ok, token, user}
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"ok": False, "error": "数据库未配置"}

    # 检查手机号是否已注册
    existing = _get_user_by_phone(phone)
    if existing:
        return {"ok": False, "error": "该手机号已注册，请直接登录"}

    pw_hash = pwd_ctx.hash(password)
    token = secrets.token_hex(24)

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


def auth_login(phone: str, password: str) -> dict:
    """
    登录。
    - 手机号不存在 → 报错
    - 密码错误 → 报错
    - 成功 → 刷新 token，返回 {ok, token, user}
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"ok": False, "error": "数据库未配置"}

    user = _get_user_by_phone(phone)
    if not user:
        return {"ok": False, "error": "手机号未注册，请先注册"}

    pw_hash = user.get("password_hash") or ""
    if not pw_hash:
        # 老用户无密码，提示重新注册或联系管理员
        return {"ok": False, "error": "该账号未设置密码，请联系管理员或重新注册"}

    if not pwd_ctx.verify(password, pw_hash):
        return {"ok": False, "error": "密码错误"}

    # 刷新 session token
    new_token = secrets.token_hex(24)
    try:
        httpx.patch(
            _url("registrations"),
            headers={**_headers(), "Prefer": "return=minimal"},
            params={"phone": f"eq.{phone}"},
            json={"session_token": new_token},
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
        },
    }
