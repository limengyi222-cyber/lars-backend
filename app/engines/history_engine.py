"""
评估历史引擎 — Supabase assessment_history 表
每次评估完成后自动保存结果，支持按用户手机号查询历史

Supabase SQL (在 SQL Editor 中执行一次):
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assessment_history (
    id            bigserial PRIMARY KEY,
    created_at    timestamptz DEFAULT now(),
    user_phone    text        DEFAULT '',
    mode          text        DEFAULT '',
    from_city     text        DEFAULT '',
    to_city       text        DEFAULT '',
    risk_value    float8,
    verdict       text        DEFAULT '',
    params        jsonb       DEFAULT '{}',
    result_summary jsonb      DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_ah_phone ON assessment_history(user_phone);
CREATE INDEX IF NOT EXISTS idx_ah_created ON assessment_history(created_at DESC);
------------------------------------------------------------
"""
import httpx
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
TABLE = "assessment_history"


def _headers(prefer_repr=False):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    h["Prefer"] = "return=representation" if prefer_repr else "return=minimal"
    return h


def save_assessment(data: dict) -> dict:
    """保存评估结果到 Supabase"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.debug("Supabase 未配置，跳过历史保存")
        return {"ok": False, "error": "Supabase not configured"}

    payload = {
        "user_phone":     data.get("phone", ""),
        "mode":           data.get("mode", ""),
        "from_city":      data.get("from_city", ""),
        "to_city":        data.get("to_city", ""),
        "risk_value":     data.get("risk_value"),          # float 或 None
        "verdict":        data.get("verdict", ""),
        "params":         data.get("params", {}),
        "result_summary": data.get("result_summary", {}),
    }

    try:
        r = httpx.post(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=_headers(),
            json=payload,
            timeout=8,
        )
        ok = r.status_code in (200, 201)
        if not ok:
            logger.warning(f"history save failed: {r.status_code} {r.text[:200]}")
        return {"ok": ok}
    except Exception as e:
        logger.warning(f"history save error: {e}")
        return {"ok": False, "error": str(e)}


def get_history(phone: str = "", limit: int = 30) -> list:
    """查询评估历史（按手机号过滤，或返回全部最近记录）"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    params = {
        "select": "*",
        "order":  "created_at.desc",
        "limit":  str(limit),
    }
    if phone:
        params["user_phone"] = f"eq.{phone}"

    try:
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=_headers(prefer_repr=True),
            params=params,
            timeout=8,
        )
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"history fetch error: {e}")
        return []


def get_stats_summary() -> dict:
    """汇总统计：各模式评估次数、PASS/FAIL 比例"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {}

    try:
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=_headers(prefer_repr=True),
            params={"select": "mode,verdict,created_at", "limit": "500", "order": "created_at.desc"},
            timeout=8,
        )
        rows = r.json()
        if not isinstance(rows, list):
            return {}

        from collections import Counter
        modes   = Counter(row.get("mode", "") for row in rows)
        verdicts = Counter(row.get("verdict", "") for row in rows)

        return {
            "total": len(rows),
            "by_mode": dict(modes),
            "by_verdict": dict(verdicts),
        }
    except Exception as e:
        logger.warning(f"stats summary error: {e}")
        return {}
