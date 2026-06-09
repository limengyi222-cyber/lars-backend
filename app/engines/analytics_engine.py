"""
数据分析引擎 — Supabase 持久化（httpx 直连 REST API）
记录：用户注册 / 评估记录 / 导出记录
提供：管理后台统计数据
"""
import os
import logging
import httpx
from typing import Dict
from datetime import date, timedelta

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


def _post(table: str, data: dict) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        r = httpx.post(_url(table), headers=_headers(), json=data, timeout=8)
        if r.status_code not in (200, 201):
            logger.warning(f"Supabase POST {table} → {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Supabase POST {table} error: {e}")
        return False


def _get(table: str, select: str = "*", order: str = None,
         limit: int = None, filters: dict = None) -> list:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    try:
        params: dict = {"select": select}
        if order:
            params["order"] = order
        if limit:
            params["limit"] = str(limit)
        if filters:
            params.update(filters)
        r = httpx.get(_url(table), headers=_headers(), params=params, timeout=8)
        if r.status_code == 200:
            return r.json() or []
        logger.warning(f"Supabase GET {table} → {r.status_code}: {r.text[:200]}")
        return []
    except Exception as e:
        logger.warning(f"Supabase GET {table} error: {e}")
        return []


def _count(table: str, filters: dict = None) -> int:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return 0
    try:
        params = {"select": "id"}
        if filters:
            params.update(filters)
        headers = {**_headers(), "Prefer": "count=exact"}
        r = httpx.get(_url(table), headers=headers, params=params, timeout=8)
        if r.status_code in (200, 206):
            cr = r.headers.get("content-range", "0/0")
            return int(cr.split("/")[-1]) if "/" in cr else len(r.json() or [])
        return 0
    except Exception as e:
        logger.warning(f"Supabase COUNT {table} error: {e}")
        return 0


# ── 写入函数 ────────────────────────────────────────────────

def log_registration(name: str, phone: str, company: str = "", ip: str = "") -> bool:
    return _post("registrations", {
        "name": name, "phone": phone, "company": company, "ip": ip
    })


def log_assessment(data: Dict) -> bool:
    return _post("assessments", {
        "phone":            data.get("phone", ""),
        "from_city":        data.get("from_city", ""),
        "to_city":          data.get("to_city", ""),
        "altitude_m":       data.get("altitude_m"),
        "route_km":         data.get("route_km"),
        "aircraft_type":    data.get("aircraft_type", ""),
        "cream_risk":       data.get("cream_risk"),
        "cream_verdict":    data.get("cream_verdict", ""),
        "terrain_verdict":  data.get("terrain_verdict", ""),
        "airspace_verdict": data.get("airspace_verdict", ""),
        "params":           data.get("params", {}),
    })


def log_export(phone: str, from_city: str, to_city: str, mode: str) -> bool:
    return _post("exports", {
        "phone": phone, "from_city": from_city,
        "to_city": to_city, "mode": mode,
    })


# ── 查询函数（管理后台）─────────────────────────────────────

def get_admin_stats() -> Dict:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"error": "数据库未配置（请检查 SUPABASE_URL / SUPABASE_KEY 环境变量）"}

    try:
        today     = date.today().isoformat()
        tomorrow  = (date.today() + timedelta(days=1)).isoformat()

        total_users       = _count("registrations")
        total_assessments = _count("assessments")
        total_exports     = _count("exports")
        today_regs        = _count("registrations", {"created_at": f"gte.{today}"})
        today_assessments = _count("assessments",   {"created_at": f"gte.{today}"})

        recent_regs = _get(
            "registrations",
            select="name,phone,company,created_at",
            order="created_at.desc", limit=20
        )
        recent_assessments = _get(
            "assessments",
            select="phone,from_city,to_city,altitude_m,aircraft_type,cream_risk,cream_verdict,terrain_verdict,airspace_verdict,created_at",
            order="created_at.desc", limit=20
        )
        recent_exports = _get(
            "exports",
            select="phone,from_city,to_city,mode,created_at",
            order="created_at.desc", limit=10
        )

        # 热门路线 Top 10
        all_a = _get("assessments", select="from_city,to_city")
        rc: Dict[str, int] = {}
        for a in all_a:
            key = f"{a.get('from_city','')}-{a.get('to_city','')}"
            rc[key] = rc.get(key, 0) + 1
        top_routes = [
            {"route": k.replace("-", " → "), "count": v}
            for k, v in sorted(rc.items(), key=lambda x: -x[1])[:10]
        ]

        # 7 天趋势（用 httpx 直接传多个同名参数）
        trend = []
        for i in range(6, -1, -1):
            d     = (date.today() - timedelta(days=i)).isoformat()
            d_nxt = (date.today() - timedelta(days=i - 1)).isoformat()
            try:
                headers = {**_headers(), "Prefer": "count=exact"}
                r = httpx.get(
                    _url("assessments"), headers=headers, timeout=8,
                    params=[("select","id"),("created_at",f"gte.{d}"),("created_at",f"lt.{d_nxt}")]
                )
                cr = r.headers.get("content-range","0/0")
                n = int(cr.split("/")[-1]) if "/" in cr else 0
            except Exception:
                n = 0
            trend.append({"date": d, "count": n})

        return {
            "total_users":           total_users,
            "total_assessments":     total_assessments,
            "total_exports":         total_exports,
            "today_regs":            today_regs,
            "today_assessments":     today_assessments,
            "recent_registrations":  recent_regs,
            "recent_assessments":    recent_assessments,
            "recent_exports":        recent_exports,
            "top_routes":            top_routes,
            "trend_7d":              trend,
        }

    except Exception as e:
        logger.error(f"get_admin_stats error: {e}")
        return {"error": str(e)}
