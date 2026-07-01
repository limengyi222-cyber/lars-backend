"""
大疆电子围栏引擎 — 航线 × DJI 限飞区比对

DJI 无人机对电子围栏区物理强制（禁飞/限高/需解锁），是"运行现实"约束：
即使航线合法适飞，落入 DJI 禁飞/限飞区，大疆机也无法起飞或被限高。

数据：dji_fence_gd.json（广东区域 446 个围栏多边形，含等级）
等级（严重度由高到低）：限飞/禁飞区 > 加强警示区 > 警示区 > 其它
判定：沿线采样点做多边形包含判断（bbox 预筛 + 射线法），聚合命中围栏与最严等级。
"""
import os, json, math, logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# 等级严重度排序
LEVEL_RANK = {"限飞/禁飞区": 3, "加强警示区": 2, "管制区": 2, "限制区": 2, "授权飞行区": 1, "警示区": 1}
LEVEL_DESC = {
    "限飞/禁飞区": "大疆机禁飞/限高，物理无法起飞（除非官方解禁）",
    "加强警示区": "进入需多次确认，强提示",
    "管制区": "管制空域，需授权",
    "限制区": "受限区，需授权",
    "授权飞行区": "需通过大疆账号解锁授权",
    "警示区": "飞行前提示，可飞但注意",
}

_ZONES = None

def _load():
    global _ZONES
    if _ZONES is not None:
        return _ZONES
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dji_fence_gd.json")
    try:
        _ZONES = json.load(open(path))
        logger.info(f"大疆围栏加载: {len(_ZONES)} 个")
    except Exception as e:
        logger.warning(f"大疆围栏加载失败: {e}")
        _ZONES = []
    return _ZONES

def _ray(lon, lat, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside

def _interp(waypoints: List[Dict], n: int = 120) -> List[Dict]:
    if len(waypoints) < 2:
        return [{"lat": waypoints[0]["lat"], "lon": waypoints[0]["lon"], "dist_km": 0.0}]
    seg = []
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
        dp = math.radians(b["lat"] - a["lat"]); dl = math.radians(b["lon"] - a["lon"])
        seg.append(6371 * 2 * math.asin(min(1, math.sqrt(math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2))))
    total = sum(seg) or 1e-9
    step = total / max(n - 1, 1)
    out = []; si = 0; cum = 0.0
    for k in range(n):
        target = k * step
        while si < len(seg) - 1 and cum + seg[si] < target - 1e-9:
            cum += seg[si]; si += 1
        t = 0.0 if seg[si] < 1e-9 else min((target - cum) / seg[si], 1.0)
        lat = waypoints[si]["lat"] + t * (waypoints[si+1]["lat"] - waypoints[si]["lat"])
        lon = waypoints[si]["lon"] + t * (waypoints[si+1]["lon"] - waypoints[si]["lon"])
        out.append({"lat": lat, "lon": lon, "dist_km": round(target, 2)})
    return out


def check_dji_fence(params: Dict) -> Dict:
    """输入: waypoints[{lat,lon}], n_samples=120"""
    zones = _load()
    wps = params["waypoints"]
    n = int(params.get("n_samples", 120))
    samples = _interp(wps, n)

    hit_names = {}      # zone name → {level, count}
    worst_rank = 0
    worst_level = ""
    restricted_pts = 0  # 命中"限飞/禁飞区"的点数
    for s in samples:
        lon, lat = s["lon"], s["lat"]
        for z in zones:
            bb = z["bbox"]
            if lon < bb[0] or lon > bb[2] or lat < bb[1] or lat > bb[3]:
                continue
            if _ray(lon, lat, z["ring"]):
                lv = z["level"]
                key = z["name"] + "|" + lv
                hit_names.setdefault(key, {"name": z["name"], "level": lv, "n": 0, "ring": z["ring"]})
                hit_names[key]["n"] += 1
                r = LEVEL_RANK.get(lv, 1)
                if r > worst_rank:
                    worst_rank = r; worst_level = lv
                if lv == "限飞/禁飞区":
                    restricted_pts += 1

    npts = len(samples) or 1
    hits = sorted(hit_names.values(), key=lambda x: -LEVEL_RANK.get(x["level"], 1))
    verdict = "CLEAR" if not hits else ("BLOCKED" if worst_rank >= 3 else "WARN")
    return {
        "verdict": verdict,               # CLEAR / WARN / BLOCKED
        "worst_level": worst_level,
        "worst_desc": LEVEL_DESC.get(worst_level, ""),
        "restricted_pct": round(restricted_pts / npts * 100, 1),
        "zones_hit": hits[:12],
        "zones_total": len(hits),
        "fence_count": len(zones),
        "note": "大疆电子围栏（DJI 机物理强制），与法定适飞相互独立；数据为大疆公开围栏，随时更新，实际以飞行时 App 为准。",
    }
