"""
数据分析引擎 — Supabase 持久化
记录：用户注册 / 评估记录 / 导出记录
提供：管理后台统计数据
"""
import os
import logging
from typing import Dict, Optional
from datetime import date, timedelta

logger = logging.getLogger(__name__)

_supabase = None


def _client():
    global _supabase
    if _supabase is not None:
        return _supabase
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        _supabase = create_client(url, key)
    except Exception as e:
        logger.error(f"Supabase init failed: {e}")
        return None
    return _supabase


# ── 写入函数 ────────────────────────────────────────

def log_registration(name: str, phone: str, company: str = "", ip: str = "") -> bool:
    c = _client()
    if not c:
        return False
    try:
        c.table("registrations").insert({
            "name": name, "phone": phone,
            "company": company, "ip": ip
        }).execute()
        return True
    except Exception as e:
        logger.warning(f"log_registration: {e}")
        return False


def log_assessment(data: Dict) -> bool:
    c = _client()
    if not c:
        return False
    try:
        c.table("assessments").insert({
            "phone":           data.get("phone", ""),
            "from_city":       data.get("from_city", ""),
            "to_city":         data.get("to_city", ""),
            "altitude_m":      data.get("altitude_m"),
            "route_km":        data.get("route_km"),
            "aircraft_type":   data.get("aircraft_type", ""),
            "cream_risk":      data.get("cream_risk"),
            "cream_verdict":   data.get("cream_verdict", ""),
            "terrain_verdict": data.get("terrain_verdict", ""),
            "airspace_verdict":data.get("airspace_verdict", ""),
            "params":          data.get("params", {}),
        }).execute()
        return True
    except Exception as e:
        logger.warning(f"log_assessment: {e}")
        return False


def log_export(phone: str, from_city: str, to_city: str, mode: str) -> bool:
    c = _client()
    if not c:
        return False
    try:
        c.table("exports").insert({
            "phone": phone, "from_city": from_city,
            "to_city": to_city, "mode": mode
        }).execute()
        return True
    except Exception as e:
        logger.warning(f"log_export: {e}")
        return False


# ── 查询函数（管理后台用）──────────────────────────

def get_admin_stats() -> Dict:
    c = _client()
    if not c:
        return {"error": "数据库未配置（请检查 SUPABASE_URL / SUPABASE_KEY 环境变量）"}

    try:
        today     = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=1)).isoformat()

        def count(table, **filters):
            q = c.table(table).select("id", count="exact")
            for k, v in filters.items():
                q = q.gte(k, v) if k == "created_at" else q.eq(k, v)
            return q.execute().count or 0

        total_users       = count("registrations")
        total_assessments = count("assessments")
        total_exports     = count("exports")
        today_regs        = count("registrations", created_at=today)
        today_assessments = count("assessments",   created_at=today)

        # 最近注册（20条）
        recent_regs = (
            c.table("registrations")
             .select("name,phone,company,created_at")
             .order("created_at", desc=True).limit(20).execute().data or []
        )

        # 最近评估（20条）
        recent_assessments = (
            c.table("assessments")
             .select("phone,from_city,to_city,altitude_m,aircraft_type,cream_risk,cream_verdict,terrain_verdict,airspace_verdict,created_at")
             .order("created_at", desc=True).limit(20).execute().data or []
        )

        # 最近导出（10条）
        recent_exports = (
            c.table("exports")
             .select("phone,from_city,to_city,mode,created_at")
             .order("created_at", desc=True).limit(10).execute().data or []
        )

        # 热门路线 Top 10
        all_a = c.table("assessments").select("from_city,to_city").execute().data or []
        rc: Dict[str, int] = {}
        for a in all_a:
            key = f"{a.get('from_city','')}-{a.get('to_city','')}"
            rc[key] = rc.get(key, 0) + 1
        top_routes = [
            {"route": k.replace("-", " → "), "count": v}
            for k, v in sorted(rc.items(), key=lambda x: -x[1])[:10]
        ]

        # 7 天趋势
        trend = []
        for i in range(6, -1, -1):
            d     = (date.today() - timedelta(days=i)).isoformat()
            d_nxt = (date.today() - timedelta(days=i - 1)).isoformat()
            n = (
                c.table("assessments").select("id", count="exact")
                 .gte("created_at", d).lt("created_at", d_nxt)
                 .execute().count or 0
            )
            trend.append({"date": d, "count": n})

        return {
            "total_users":        total_users,
            "total_assessments":  total_assessments,
            "total_exports":      total_exports,
            "today_regs":         today_regs,
            "today_assessments":  today_assessments,
            "recent_registrations": recent_regs,
            "recent_assessments":   recent_assessments,
            "recent_exports":       recent_exports,
            "top_routes":           top_routes,
            "trend_7d":             trend,
        }

    except Exception as e:
        logger.error(f"get_admin_stats: {e}")
        return {"error": str(e)}
